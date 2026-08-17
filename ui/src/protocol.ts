export { PROTOCOL_VERSION, type EventType, type RequestType } from "./protocol.generated.js";
import { PROTOCOL_VERSION, type EventType, type RequestType } from "./protocol.generated.js";

export interface ProtocolMessage {
  protocolVersion: typeof PROTOCOL_VERSION;
  type: RequestType | EventType;
  id?: string;
  requestId?: string;
  callId?: string;
  text?: string;
  approved?: boolean;
  width?: number;
  height?: number;
  imagePaths?: string[];
  payload?: Record<string, unknown>;
}

export function decodeMessage(line: string): ProtocolMessage {
  const value: unknown = JSON.parse(line);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Backend protocol message must be an object");
  }
  const message = value as Partial<ProtocolMessage>;
  if (message.protocolVersion !== PROTOCOL_VERSION) {
    throw new Error(`Protocol mismatch: backend=${String(message.protocolVersion)}, frontend=${PROTOCOL_VERSION}`);
  }
  if (typeof message.type !== "string" || !message.type) {
    throw new Error("Backend protocol message has no type");
  }
  return message as ProtocolMessage;
}
