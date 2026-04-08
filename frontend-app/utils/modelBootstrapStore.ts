export type ModelBootstrapPhase =
  | "warming"
  | "training"
  | "ready"
  | "failed";

export interface ModelBootstrapState {
  phase: ModelBootstrapPhase;
  message: string;
  updatedAt: number;
}

let state: ModelBootstrapState = {
  phase: "warming",
  message: "Collecting baseline behavior...",
  updatedAt: Date.now(),
};

const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((listener) => listener());
}

export function subscribeToModelBootstrapStore(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getModelBootstrapState() {
  return state;
}

export function setModelBootstrapState(next: Omit<ModelBootstrapState, "updatedAt">) {
  state = {
    ...next,
    updatedAt: Date.now(),
  };
  emit();
}

export function isModelReadyForScoring() {
  return state.phase === "ready";
}
