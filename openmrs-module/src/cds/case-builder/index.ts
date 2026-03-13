export { fetchPatient, fetchEncounters } from './openmrs-api';
export { buildCaseXml, buildModelPrompt } from './case-xml';
export { generateCds, generateCdsStreaming, checkCdsHealth } from './cds-api';
export type { CdsResult, KbHit, KbQueryResult, StreamEvent } from './cds-api';
