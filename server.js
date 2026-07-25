// server.js
// Agentic OCR -> translate pipeline.
// "Agentic" bit: after OCR, the pipeline evaluates its own confidence and
// DECIDES whether to retry with a stricter prompt before moving on to
// translation. That decision + both attempts are captured as distinct spans,
// so the trace in SigNoz visibly shows the agent branching, not just a
// fixed A->B->C call chain.

require('dotenv').config();

const express = require('express');
const multer = require('multer');
const OpenAI = require('openai');
const { trace, metrics, SpanStatusCode } = require('@opentelemetry/api');

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

// Groq is OpenAI-compatible: same client, different base URL + key.
const client = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: 'https://api.groq.com/openai/v1',
});
const MODEL_NAME = 'qwen/qwen3.6-27b'; // Groq's only current vision-capable model

// --- OpenTelemetry handles -------------------------------------------------
const tracer = trace.getTracer('ocr-translate-agent');
const meter = metrics.getMeter('ocr-translate-agent');

const tokenUsageCounter = meter.createCounter('gemini.tokens.total', {
  description: 'Total Gemini tokens consumed, tagged by pipeline stage',
});
const retryCounter = meter.createCounter('pipeline.ocr.retries', {
  description: 'Number of times the OCR stage decided to retry due to low confidence',
});
const pipelineDuration = meter.createHistogram('pipeline.duration.ms', {
  description: 'End-to-end pipeline duration in milliseconds',
  unit: 'ms',
});

const CONFIDENCE_THRESHOLD = Number(process.env.CONFIDENCE_THRESHOLD || 0.7);

// --- Groq vision calls, each wrapped in its own span ------------------------

async function runOcr(imageBase64, mimeType, { strict }) {
  return tracer.startActiveSpan(strict ? 'ocr.attempt.strict' : 'ocr.attempt.initial', async (span) => {
    span.setAttribute('ocr.strict_mode', strict);
    try {
      const prompt = strict
        ? 'Extract ALL text from this image exactly as written, preserving line breaks. ' +
          'Be extremely careful with numbers, names, and small text. ' +
          'Respond ONLY as JSON: {"text": "...", "confidence": 0.0-1.0}'
        : 'Extract all text from this image. ' +
          'Respond ONLY as JSON: {"text": "...", "confidence": 0.0-1.0}';

      const completion = await client.chat.completions.create({
        model: MODEL_NAME,
        response_format: { type: 'json_object' },
        reasoning_effort: 'none',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: `data:${mimeType};base64,${imageBase64}` } },
            ],
          },
        ],
      });

      const rawText = completion.choices[0].message.content;
      const responseText = rawText.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
      const usage = completion.usage || {};
      tokenUsageCounter.add(usage.total_tokens || 0, { stage: 'ocr', strict: String(strict) });
      span.setAttribute('gemini.tokens', usage.total_tokens || 0);

      let parsed;
      try {
        parsed = JSON.parse(responseText.replace(/```json|```/g, '').trim());
      } catch {
        parsed = { text: responseText, confidence: 0.5 }; // model didn't follow format
        span.addEvent('ocr.parse_fallback', { raw_length: responseText.length });
      }

      span.setAttribute('ocr.confidence', parsed.confidence ?? 0);
      span.setStatus({ code: SpanStatusCode.OK });
      return parsed;
    } catch (err) {
      span.recordException(err);
      span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
      throw err;
    } finally {
      span.end();
    }
  });
}

async function runTranslate(text, targetLang) {
  return tracer.startActiveSpan('translate.attempt', async (span) => {
    span.setAttribute('translate.target_lang', targetLang);
    try {
      const completion = await client.chat.completions.create({
        model: MODEL_NAME,
        reasoning_effort: 'none',
        messages: [
          {
            role: 'user',
            content: `Translate the following text to ${targetLang}. Return ONLY the translation, no preamble:\n\n${text}`,
          },
        ],
      });
      const rawTranslated = completion.choices[0].message.content;
      const translated = rawTranslated.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
      const usage = completion.usage || {};
      tokenUsageCounter.add(usage.total_tokens || 0, { stage: 'translate' });
      span.setAttribute('gemini.tokens', usage.total_tokens || 0);
      span.setStatus({ code: SpanStatusCode.OK });
      return translated;
    } catch (err) {
      span.recordException(err);
      span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
      throw err;
    } finally {
      span.end();
    }
  });
}

// --- The agent: decides whether to retry, then translates ------------------

async function runPipeline(imageBase64, mimeType, targetLang) {
  return tracer.startActiveSpan('pipeline.ocr_translate', async (pipelineSpan) => {
    const startedAt = Date.now();
    try {
      let ocrResult = await runOcr(imageBase64, mimeType, { strict: false });

      // --- THE AGENTIC DECISION ---
      const lowConfidence = (ocrResult.confidence ?? 0) < CONFIDENCE_THRESHOLD;
      pipelineSpan.setAttribute('pipeline.decision.retry_triggered', lowConfidence);
      pipelineSpan.addEvent('pipeline.decision', {
        initial_confidence: ocrResult.confidence ?? 0,
        threshold: CONFIDENCE_THRESHOLD,
        decision: lowConfidence ? 'retry_with_strict_prompt' : 'proceed_to_translation',
      });

      if (lowConfidence) {
        retryCounter.add(1, { reason: 'low_confidence' });
        ocrResult = await runOcr(imageBase64, mimeType, { strict: true });
      }

      const translated = await runTranslate(ocrResult.text, targetLang);

      pipelineSpan.setStatus({ code: SpanStatusCode.OK });
      return {
        extractedText: ocrResult.text,
        confidence: ocrResult.confidence,
        retried: lowConfidence,
        translatedText: translated,
      };
    } catch (err) {
      pipelineSpan.recordException(err);
      pipelineSpan.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
      throw err;
    } finally {
      pipelineDuration.record(Date.now() - startedAt);
      pipelineSpan.end();
    }
  });
}

// --- SRE Sidekick: answers questions about the pipeline by querying SigNoz ---

const SIGNOZ_URL = process.env.SIGNOZ_URL || 'http://localhost:8080';
const SIGNOZ_API_KEY = process.env.SIGNOZ_API_KEY;

// Picks a query template based on keywords in the question. Not a general NL
// interface - a small set of reliable patterns is more honest for a hackathon
// demo than pretending to understand arbitrary questions.
function pickTemplate(question) {
  const q = question.toLowerCase();
  if (/(error|fail|broke|broken)/.test(q)) return 'errors';
  if (/(retry|retries|retried)/.test(q)) return 'retries';
  if (/(token|cost|expensive)/.test(q)) return 'tokens';
  if (/(latency|slow|duration|speed|fast)/.test(q)) return 'latency';
  return 'requests';
}

function buildQuery(template, sinceMs, nowMs) {
  const base = { start: sinceMs, end: nowMs, requestType: 'time_series' };
  switch (template) {
    case 'errors':
      return {
        ...base,
        compositeQuery: {
          queries: [{
            type: 'builder_query',
            spec: {
              name: 'A',
              signal: 'traces',
              aggregations: [{ expression: 'count()' }],
              filter: { expression: "service.name = 'ocr-translate-agent' AND name = 'pipeline.ocr_translate' AND status_code = 'STATUS_CODE_ERROR'" },
              stepInterval: 60,
              disabled: false,
            },
          }],
        },
      };
    case 'retries':
      return {
        ...base,
        compositeQuery: {
          queries: [{
            type: 'builder_query',
            spec: {
              name: 'A',
              signal: 'metrics',
              aggregations: [{ metricName: 'pipeline.ocr.retries', timeAggregation: 'increase', spaceAggregation: 'sum' }],
              stepInterval: 60,
              disabled: false,
            },
          }],
        },
      };
    case 'tokens':
      return {
        ...base,
        compositeQuery: {
          queries: [{
            type: 'builder_query',
            spec: {
              name: 'A',
              signal: 'metrics',
              aggregations: [{ metricName: 'gemini.tokens.total', timeAggregation: 'increase', spaceAggregation: 'sum' }],
              groupBy: [{ name: 'stage' }],
              stepInterval: 60,
              disabled: false,
            },
          }],
        },
      };
    case 'latency':
      return {
        ...base,
        compositeQuery: {
          queries: [{
            type: 'builder_query',
            spec: {
              name: 'A',
              signal: 'metrics',
              aggregations: [{ metricName: 'pipeline.duration.ms.max', timeAggregation: 'avg', spaceAggregation: 'avg' }],
              stepInterval: 60,
              disabled: false,
            },
          }],
        },
      };
    default: // requests
      return {
        ...base,
        compositeQuery: {
          queries: [{
            type: 'builder_query',
            spec: {
              name: 'A',
              signal: 'traces',
              aggregations: [{ expression: 'count()' }],
              filter: { expression: "service.name = 'ocr-translate-agent' AND name = 'pipeline.ocr_translate'" },
              stepInterval: 60,
              disabled: false,
            },
          }],
        },
      };
  }
}

async function querySignoz(body) {
  const resp = await fetch(`${SIGNOZ_URL}/api/v5/query_range`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'SIGNOZ-API-KEY': SIGNOZ_API_KEY,
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`SigNoz query failed (${resp.status}): ${text}`);
  }
  return resp.json();
}

app.post('/sidekick', express.json(), async (req, res) => {
  return tracer.startActiveSpan('sidekick.ask', async (span) => {
    const question = (req.body && req.body.question || '').trim();
    if (!question) {
      span.end();
      return res.status(400).json({ error: 'body must include "question"' });
    }
    span.setAttribute('sidekick.question', question);

    try {
      const template = pickTemplate(question);
      span.setAttribute('sidekick.template', template);

      const nowMs = Date.now();
      const sinceMs = nowMs - 60 * 60 * 1000; // last 1 hour
      const query = buildQuery(template, sinceMs, nowMs);

      const raw = await querySignoz(query);

      // Summarize the raw SigNoz JSON into a plain-English sentence via Groq.
      const completion = await client.chat.completions.create({
        model: MODEL_NAME,
        reasoning_effort: 'none',
        messages: [{
          role: 'user',
          content:
            `You are an SRE assistant. A user asked: "${question}"\n` +
            `Here is the raw observability query result (JSON) for the last hour:\n${JSON.stringify(raw)}\n\n` +
            `Answer the user's question in ONE short, plain-English sentence based only on this data. ` +
            `If the data is empty or you can't tell, say so honestly rather than guessing a number.`,
        }],
      });
      const rawAnswer = completion.choices[0].message.content;
      const answer = rawAnswer.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

      span.setStatus({ code: SpanStatusCode.OK });
      res.json({ question, template, answer });
    } catch (err) {
      span.recordException(err);
      span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
      console.error('[sidekick error]', err.message);
      res.status(500).json({ error: 'sidekick failed', detail: err.message });
    } finally {
      span.end();
    }
  });
});

// --- HTTP layer --------------------------------------------------------------

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

app.post('/ocr-translate', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'multipart field "image" is required' });
  }
  const targetLang = req.body.targetLang || 'English';

  try {
    const result = await runPipeline(
      req.file.buffer.toString('base64'),
      req.file.mimetype,
      targetLang
    );
    res.json(result);
  } catch (err) {
    console.error('[pipeline error]', err.message);
    res.status(500).json({ error: 'pipeline failed', detail: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`ocr-translate-agent listening on :${PORT}`);
});
