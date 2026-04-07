import React from "react";
import { View } from "react-native";
import { recordTouchSnapshot } from "../hooks/useBehaviorCsvCapture";
import { recordContinuousTouchSnapshot } from "../utils/continuousModelBuffer";

type TouchNativeEvent = {
  locationX: number;
  locationY: number;
  pageX: number;
  pageY: number;
  force?: number;
};

export default function GestureWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <View
      style={{ flex: 1 }}
      onTouchStart={(evt) => {
        const nativeEvent = evt.nativeEvent as TouchNativeEvent;
        recordTouchSnapshot({
          action: "start",
          touchX: nativeEvent.locationX,
          touchY: nativeEvent.locationY,
          pageX: nativeEvent.pageX,
          pageY: nativeEvent.pageY,
        });
        recordContinuousTouchSnapshot({
          action: "start",
          pageX: nativeEvent.pageX,
          pageY: nativeEvent.pageY,
          pressure: nativeEvent.force,
        });
      }}
      onTouchMove={(evt) => {
        const nativeEvent = evt.nativeEvent as TouchNativeEvent;
        recordTouchSnapshot({
          action: "move",
          touchX: nativeEvent.locationX,
          touchY: nativeEvent.locationY,
          pageX: nativeEvent.pageX,
          pageY: nativeEvent.pageY,
        });
        recordContinuousTouchSnapshot({
          action: "move",
          pageX: nativeEvent.pageX,
          pageY: nativeEvent.pageY,
          pressure: nativeEvent.force,
        });
      }}
      onTouchEnd={(evt) => {
        const nativeEvent = evt.nativeEvent as TouchNativeEvent;
        recordTouchSnapshot({
          action: "end",
          touchX: nativeEvent.locationX,
          touchY: nativeEvent.locationY,
          pageX: nativeEvent.pageX,
          pageY: nativeEvent.pageY,
        });
        recordContinuousTouchSnapshot({
          action: "end",
          pageX: nativeEvent.pageX,
          pageY: nativeEvent.pageY,
          pressure: nativeEvent.force,
        });
      }}
    >
      {children}
    </View>
  );
}
