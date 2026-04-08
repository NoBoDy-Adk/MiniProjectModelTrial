import { Slot } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import ModelConfidenceOverlay from "../components/ModelConfidenceOverlay";
import GestureWrapper from "../components/GestureWrapper";
import useContinuousModelScoring from "../hooks/useContinuousModelScoring";
import useBehaviorCsvCapture from "../hooks/useBehaviorCsvCapture";
import useContinuousTempInputSync from "../hooks/useContinuousTempInputSync";
import useModelColdStartBootstrap from "../hooks/useModelColdStartBootstrap";

export default function Layout() {
  useBehaviorCsvCapture();
  useContinuousTempInputSync();
  useModelColdStartBootstrap();
  useContinuousModelScoring();

  return (
    <GestureWrapper>
      <SafeAreaView style={{ flex: 1 }}>
        <Slot />
        <ModelConfidenceOverlay />
      </SafeAreaView>
    </GestureWrapper>
  );
}
