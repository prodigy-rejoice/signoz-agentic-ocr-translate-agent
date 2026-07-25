// instrumentation.js
// Loaded FIRST (via `node -r ./instrumentation.js server.js`) so auto-instrumentation
// can patch http/express/etc before the app code requires them.
//
// Sends traces + metrics to a local SigNoz OTel Collector over OTLP/HTTP.
// Default collector endpoint for a self-hosted SigNoz via docker compose is
// http://localhost:4318 (HTTP) - adjust via .env if yours differs.

require('dotenv').config();

const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { OTLPTraceExporter } = require('@opentelemetry/exporter-trace-otlp-http');
const { OTLPMetricExporter } = require('@opentelemetry/exporter-metrics-otlp-http');
const { PeriodicExportingMetricReader } = require('@opentelemetry/sdk-metrics');
const resourcesPkg = require('@opentelemetry/resources');
const resourceFromAttributes = resourcesPkg.resourceFromAttributes
  || ((attrs) => new resourcesPkg.Resource(attrs));
const { ATTR_SERVICE_NAME, ATTR_SERVICE_VERSION } = require('@opentelemetry/semantic-conventions');

const OTLP_ENDPOINT = process.env.OTLP_ENDPOINT || 'http://localhost:4318';

const sdk = new NodeSDK({
  resource: resourceFromAttributes({
    [ATTR_SERVICE_NAME]: process.env.OTEL_SERVICE_NAME || 'ocr-translate-agent',
    [ATTR_SERVICE_VERSION]: '1.0.0',
  }),
  traceExporter: new OTLPTraceExporter({
    url: `${OTLP_ENDPOINT}/v1/traces`,
  }),
  metricReader: new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({
      url: `${OTLP_ENDPOINT}/v1/metrics`,
    }),
    exportIntervalMillis: 5000,
  }),
  instrumentations: [getNodeAutoInstrumentations()],
});

sdk.start();

process.on('SIGTERM', () => {
  sdk.shutdown().finally(() => process.exit(0));
});

console.log(`[otel] instrumentation started, exporting to ${OTLP_ENDPOINT}`);
