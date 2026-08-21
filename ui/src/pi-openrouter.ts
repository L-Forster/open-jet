import type { AuthEvent, AuthPrompt, Credential } from "@earendil-works/pi-ai";
import {
  ExtensionSelectorComponent,
  LoginDialogComponent,
  ModelRuntime,
  ModelSelectorComponent,
  SettingsManager,
  initTheme,
  readStoredCredential,
} from "@earendil-works/pi-coding-agent";
import type { Container, Editor, TUI } from "@earendil-works/pi-tui";

let themeReady = false;
let runtimePromise: Promise<ModelRuntime> | undefined;

export function ensurePiTheme(): void {
  if (themeReady) return;
  initTheme();
  themeReady = true;
}

export function apiKeyFromCredential(credential: Credential | undefined): string | undefined {
  if (!credential) return undefined;
  if (credential.type === "api_key") return credential.key?.trim() || undefined;
  if (credential.type === "oauth") return credential.access?.trim() || undefined;
  return undefined;
}

export function storedOpenRouterApiKey(): string | undefined {
  return apiKeyFromCredential(readStoredCredential("openrouter"));
}

export async function openRouterRuntime(): Promise<ModelRuntime> {
  runtimePromise ??= ModelRuntime.create({
    modelsPath: null,
    allowModelNetwork: true,
    refreshOnCreate: true,
  }).catch((error) => {
    // Don't cache a rejected create (e.g. offline at startup) — the next
    // /login or /cloud should retry instead of replaying the failure forever.
    runtimePromise = undefined;
    throw error;
  });
  return runtimePromise;
}

export async function loginOpenRouter(options: {
  tui: TUI;
  editorContainer: Container;
  editor: Editor;
}): Promise<string | undefined> {
  ensurePiTheme();
  const authType = await pickAuthType(options);
  if (!authType) return undefined;
  const runtime = await openRouterRuntime();
  const credential = await runLoginDialog(options, runtime, authType);
  return apiKeyFromCredential(credential);
}

export function pickOpenRouterModel(options: {
  tui: TUI;
  editorContainer: Container;
  editor: Editor;
  cwd: string;
  currentModelId?: string;
}): Promise<string | undefined> {
  ensurePiTheme();
  return new Promise((resolve, reject) => {
    void (async () => {
      const runtime = await openRouterRuntime();
      const settings = SettingsManager.create(options.cwd);
      const scoped = runtime.getModels("openrouter").map((model) => ({ model }));
      const current = scoped.find((item) => item.model.id === options.currentModelId)?.model;
      const selector = new ModelSelectorComponent(
        options.tui,
        current,
        settings,
        runtime,
        scoped,
        (model) => {
          selector.dispose();
          restoreEditor(options);
          resolve(model.id);
        },
        () => {
          selector.dispose();
          restoreEditor(options);
          resolve(undefined);
        },
      );
      options.editorContainer.clear();
      options.editorContainer.addChild(selector);
      options.tui.setFocus(selector);
      options.tui.requestRender();
    })().catch(reject);
  });
}

function restoreEditor(options: { tui: TUI; editorContainer: Container; editor: Editor }): void {
  options.editorContainer.clear();
  options.editorContainer.addChild(options.editor);
  options.tui.setFocus(options.editor);
  options.tui.requestRender();
}

function pickAuthType(options: {
  tui: TUI;
  editorContainer: Container;
  editor: Editor;
}): Promise<"oauth" | "api_key" | undefined> {
  const oauthLabel = "Sign in with OpenRouter";
  const apiKeyLabel = "Sign in with an API key";
  return new Promise((resolve) => {
    const selector = new ExtensionSelectorComponent(
      "Select authentication method for OpenRouter:",
      [oauthLabel, apiKeyLabel],
      (option) => {
        restoreEditor(options);
        resolve(option === oauthLabel ? "oauth" : "api_key");
      },
      () => {
        restoreEditor(options);
        resolve(undefined);
      },
    );
    options.editorContainer.clear();
    options.editorContainer.addChild(selector);
    options.tui.setFocus(selector);
    options.tui.requestRender();
  });
}

async function runLoginDialog(
  options: { tui: TUI; editorContainer: Container; editor: Editor },
  runtime: ModelRuntime,
  authType: "oauth" | "api_key",
): Promise<Credential | undefined> {
  const dialog = new LoginDialogComponent(options.tui, "openrouter", () => {}, "OpenRouter");
  options.editorContainer.clear();
  options.editorContainer.addChild(dialog);
  options.tui.setFocus(dialog);
  options.tui.requestRender();
  try {
    return await runtime.login("openrouter", authType, {
      signal: dialog.signal,
      prompt: (prompt) => showAuthPrompt(options, dialog, prompt),
      notify: (event) => notifyAuthDialog(dialog, event),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message === "Login cancelled") return undefined;
    throw error;
  } finally {
    restoreEditor(options);
  }
}

function notifyAuthDialog(dialog: LoginDialogComponent, event: AuthEvent): void {
  if (event.type === "auth_url") dialog.showAuth(event.url, event.instructions);
  else if (event.type === "device_code") {
    dialog.showDeviceCode(event);
    dialog.showWaiting("Waiting for authentication...");
  } else if (event.type === "info") dialog.showInfo(event.message, event.links);
  else dialog.showProgress(event.message);
}

function showAuthPrompt(
  options: { tui: TUI; editorContainer: Container; editor: Editor },
  dialog: LoginDialogComponent,
  prompt: AuthPrompt,
): Promise<string> {
  if (prompt.type === "select") {
    return new Promise((resolve, reject) => {
      const restoreDialog = () => {
        options.editorContainer.clear();
        options.editorContainer.addChild(dialog);
        options.tui.setFocus(dialog);
        options.tui.requestRender();
      };
      const labels = prompt.options.map((option) => option.label);
      const selector = new ExtensionSelectorComponent(prompt.message, labels, (optionLabel) => {
        restoreDialog();
        const id = prompt.options.find((option) => option.label === optionLabel)?.id;
        if (id) resolve(id);
        else reject(new Error("Login cancelled"));
      }, () => {
        restoreDialog();
        reject(new Error("Login cancelled"));
      });
      options.editorContainer.clear();
      options.editorContainer.addChild(selector);
      options.tui.setFocus(selector);
      options.tui.requestRender();
    });
  }
  if (prompt.type === "manual_code") return dialog.showManualInput(prompt.message);
  return dialog.showPrompt(prompt.message, prompt.placeholder);
}
