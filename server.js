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

// --- HTTP layer --------------------------------------------------------------

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
