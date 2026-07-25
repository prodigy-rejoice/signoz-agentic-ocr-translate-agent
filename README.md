# ocr-translate-agent

An agentic OCR -> translate pipeline (Gemini) fully instrumented with
OpenTelemetry and observed via SigNoz.

**The agentic part:** after the first OCR pass, the pipeline checks its own
confidence score. If it's below `CONFIDENCE_THRESHOLD`, it *decides* to retry
with a stricter prompt before moving on to translation. Both the decision and
both OCR attempts show up as distinct spans in the trace, so you can see the
agent branching in the SigNoz UI - not just a fixed call chain.

## 1. Start SigNoz (self-hosted, local)

Do this in a separate terminal/folder, NOT inside this project:

```bash
git clone -b main https://github.com/SigNoz/signoz.git
cd signoz/deploy/docker
docker compose up -d
```

Give it a minute to start, then open http://localhost:3301 to confirm the
SigNoz UI loads. The OTel Collector will be listening on localhost:4318
(HTTP) - that's what this backend sends to.

If your local ports differ, update OTLP_ENDPOINT in your `.env`.

## 2. Set up this backend

```bash
cd signoz-hackathon-backend
npm install
cp .env.example .env
# edit .env and paste in your GEMINI_API_KEY
npm start
```

You should see:
```
[otel] instrumentation started, exporting to http://localhost:4318
ocr-translate-agent listening on :3000
```

## 3. Send a test request

```bash
curl -X POST http://localhost:3000/ocr-translate \
  -F "image=@/path/to/some/image.jpg" \
  -F "targetLang=Spanish"
```

Then go to the SigNoz UI -> Traces, and look for the `ocr-translate-agent`
service. You should see a `pipeline.ocr_translate` root span containing
`ocr.attempt.initial`, optionally `ocr.attempt.strict` if a retry fired, and
`translate.attempt`.

## 4. To force a retry (for a good demo trace)

Send a blurry/low-quality image, or a very cluttered one - Gemini's
self-reported confidence will often drop below the 0.7 threshold and you'll
see the retry span appear.

## What's instrumented

- **Traces**: full pipeline span tree, including the branch decision as a
  span event on the root span (`pipeline.decision`)
- **Metrics**: `gemini.tokens.total` (tagged by stage: ocr/translate),
  `pipeline.ocr.retries`, `pipeline.duration.ms`
- **Logs**: errors are logged to stdout and recorded as exceptions on their
  span (`span.recordException`), so they correlate to the trace in SigNoz

## Next steps (not yet built)

- SigNoz dashboard: latency per stage, token cost over time, retry rate
- One alert (e.g. p95 pipeline duration or error rate)
- Flutter "SRE Sidekick" that queries SigNoz to answer questions about the
  pipeline in natural language
