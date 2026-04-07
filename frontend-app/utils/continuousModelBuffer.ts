export interface ContinuousModelEvent {
  accelX: number;
  accelY: number;
  touchPressure: number;
  duration: number;
}

const MAX_EVENTS = 500;
const listeners = new Set<() => void>();
let events: ContinuousModelEvent[] = [];
let gestureStartTimestamp: number | null = null;
let totalSamples = 0;

function emitChange() {
  listeners.forEach((listener) => listener());
}

export function subscribeToContinuousModelBuffer(listener: () => void) {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
}

export function getContinuousModelEvents() {
  return events;
}

export function getContinuousModelTotalSamples() {
  return totalSamples;
}

export function recordContinuousTouchSnapshot(payload: {
  action: "start" | "move" | "end";
  pageX: number;
  pageY: number;
  pressure?: number;
}) {
  const now = Date.now();

  if (payload.action === "start" || gestureStartTimestamp === null) {
    gestureStartTimestamp = now;
  }

  const duration = Math.max(1, now - gestureStartTimestamp);

  events = [
    ...events.slice(-(MAX_EVENTS - 1)),
    {
      accelX: payload.pageX,
      accelY: payload.pageY,
      touchPressure: payload.pressure ?? 0.5,
      duration,
    },
  ];
  totalSamples += 1;

  if (payload.action === "end") {
    gestureStartTimestamp = null;
  }

  emitChange();
}
