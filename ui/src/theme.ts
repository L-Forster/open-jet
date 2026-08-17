import chalk from "chalk";
import type { EditorTheme, MarkdownTheme, SelectListTheme } from "@earendil-works/pi-tui";

export const palette = {
  green: "#63b483",
  greenSoft: "#94c3a4",
  greenBright: "#79bd91",
  greenPale: "#b9d4c1",
  background: "#0b0f0d",
  surface: "#101715",
  surfaceRaised: "#121816",
  userSurface: "#292c2b",
  border: "#4a514d",
  dim: "#8a928c",
  muted: "#d1d5db",
  warning: "#facc15",
  error: "#ef4444",
  errorSoft: "#fca5a5",
} as const;

export const selectTheme: SelectListTheme = {
  selectedPrefix: (text) => chalk.hex(palette.green).bold(text),
  selectedText: (text) => chalk.hex(palette.greenPale).bold(text),
  description: (text) => chalk.hex(palette.dim)(text),
  scrollInfo: (text) => chalk.hex(palette.dim)(text),
  noMatch: (text) => chalk.hex(palette.warning)(text),
};

export const editorTheme: EditorTheme = {
  borderColor: (text) => chalk.hex(palette.border)(text),
  selectList: selectTheme,
};

export const markdownTheme: MarkdownTheme = {
  heading: (text) => chalk.hex(palette.greenBright).bold(text),
  link: (text) => chalk.hex(palette.greenSoft)(text),
  linkUrl: (text) => chalk.hex(palette.dim)(text),
  code: (text) => chalk.hex(palette.greenPale)(text),
  codeBlock: (text) => chalk.hex("#dbe7ee")(text),
  codeBlockBorder: (text) => chalk.hex(palette.border)(text),
  quote: (text) => chalk.hex(palette.muted).italic(text),
  quoteBorder: (text) => chalk.hex(palette.green)(text),
  hr: (text) => chalk.hex(palette.border)(text),
  listBullet: (text) => chalk.hex(palette.green)(text),
  bold: (text) => chalk.bold(text),
  italic: (text) => chalk.italic(text),
  strikethrough: (text) => chalk.strikethrough(text),
  underline: (text) => chalk.underline(text),
};
