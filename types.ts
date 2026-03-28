
export interface LayerConfig {
  id: string;
  neurons: number;
  type: 'input' | 'hidden' | 'output';
}

export interface Neuron {
  id: string;
  layerIndex: number;
  neuronIndex: number;
  value: number;
}

export interface Connection {
  sourceId: string;
  targetId: string;
  weight: number;
}

export interface TrainingPoint {
  input: number[];
  output: number[];
}

export interface TrainingStats {
  epoch: number;
  loss: number;
}
