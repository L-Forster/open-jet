import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { PROTOCOL_VERSION, decodeMessage, type ProtocolMessage, type RequestType } from "./protocol.js";

export type MessageListener = (message: ProtocolMessage) => void;

export class OpenJetRpcClient {
  private child?: ChildProcessWithoutNullStreams;
  private listeners = new Set<MessageListener>();
  private nextId = 1;
  private stderrTail = "";
  private pending = new Map<string, { resolve: (message: ProtocolMessage) => void; reject: (error: Error) => void }>();

  onMessage(listener: MessageListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  start(): void {
    if (this.child) return;
    const python = process.env.OPENJET_PYTHON;
    if (!python) throw new Error("OPENJET_PYTHON is missing; launch the TUI through the openjet command.");
    const backendArgs = ["-m", "src.tui_server"];
    if (process.env.OPENJET_FORCE_SETUP === "1") backendArgs.push("--force-setup");
    const wslDistribution = process.env.OPENJET_WSL_DISTRO;
    const wslCwd = process.env.OPENJET_WSL_CWD;
    const executable = wslDistribution && wslCwd ? "wsl.exe" : python;
    const args = wslDistribution && wslCwd
      ? ["-d", wslDistribution, "--cd", wslCwd, "--", python, ...backendArgs]
      : backendArgs;
    this.child = spawn(executable, args, { cwd: process.cwd(), env: process.env, stdio: ["pipe", "pipe", "pipe"] });
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk: string) => {
      this.stderrTail = (this.stderrTail + chunk).slice(-4000);
    });
    createInterface({ input: this.child.stdout }).on("line", (line) => {
      try {
        const message = decodeMessage(line);
        if (message.requestId && this.pending.has(message.requestId)) {
          const waiter = this.pending.get(message.requestId)!;
          this.pending.delete(message.requestId);
          if (message.type === "error") waiter.reject(new Error(message.text ?? "OpenJet service request failed."));
          else waiter.resolve(message);
          return;
        }
        for (const listener of this.listeners) listener(message);
      } catch (error) {
        this.emitLocalError(error instanceof Error ? error.message : String(error));
      }
    });
    this.child.on("error", (error) => this.emitLocalError(`Backend failed to start: ${error.message}`, true));
    this.child.on("exit", (code, signal) => {
      if (code === 0) return;
      const detail = this.stderrTail.trim();
      this.emitLocalError(
        `Backend exited (${signal ?? code ?? "unknown"})${detail ? `: ${detail}` : ""}`,
        true,
      );
    });
  }

  request(type: RequestType, fields: Omit<ProtocolMessage, "protocolVersion" | "type" | "id"> = {}): string {
    if (!this.child?.stdin.writable) throw new Error("OpenJet backend is not connected.");
    const id = `ui-${this.nextId++}`;
    const message: ProtocolMessage = { protocolVersion: PROTOCOL_VERSION, type, id, ...fields };
    this.child.stdin.write(`${JSON.stringify(message)}\n`);
    return id;
  }

  call(
    type: RequestType,
    fields: Omit<ProtocolMessage, "protocolVersion" | "type" | "id"> = {},
    signal?: AbortSignal,
  ): Promise<ProtocolMessage> {
    const id = this.request(type, fields);
    return new Promise<ProtocolMessage>((resolve, reject) => {
      const abort = () => {
        this.pending.delete(id);
        reject(new Error(`OpenJet ${type} request was aborted.`));
      };
      if (signal?.aborted) return abort();
      this.pending.set(id, {
        resolve: (message) => {
          signal?.removeEventListener("abort", abort);
          resolve(message);
        },
        reject: (error) => {
          signal?.removeEventListener("abort", abort);
          reject(error);
        },
      });
      signal?.addEventListener("abort", abort, { once: true });
    });
  }

  async close(): Promise<void> {
    if (!this.child) return;
    if (this.child.stdin.writable) this.request("shutdown");
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        this.child?.kill();
        resolve();
      }, 1500);
      this.child?.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
    this.child = undefined;
    for (const waiter of this.pending.values()) waiter.reject(new Error("OpenJet backend closed."));
    this.pending.clear();
  }

  private emitLocalError(text: string, fatal = false): void {
    const message: ProtocolMessage = {
      protocolVersion: PROTOCOL_VERSION,
      type: "error",
      text,
      payload: { fatal },
    };
    for (const listener of this.listeners) listener(message);
  }
}
