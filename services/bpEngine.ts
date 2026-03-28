
/**
 * 一个轻量级的BP神经网络实现，支持动态层级和训练可视化
 */

export class BPNetwork {
  layers: number[];
  weights: number[][][]; // [layer][target][source]
  biases: number[][];    // [layer][neuron]
  learningRate: number;

  constructor(layers: number[], learningRate: number = 0.1) {
    this.layers = layers;
    this.learningRate = learningRate;
    this.weights = [];
    this.biases = [];

    // 初始化权重和偏置
    for (let i = 0; i < layers.length - 1; i++) {
      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];
      for (let j = 0; j < layers[i + 1]; j++) {
        const neuronWeights: number[] = [];
        for (let k = 0; k < layers[i]; k++) {
          neuronWeights.push(Math.random() * 2 - 1); // -1 to 1
        }
        layerWeights.push(neuronWeights);
        layerBiases.push(Math.random() * 2 - 1);
      }
      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }
  }

  sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  sigmoidDeriv(x: number): number {
    const s = this.sigmoid(x);
    return s * (1 - s);
  }

  feedForward(inputs: number[]): number[][] {
    let current = inputs;
    const activations = [inputs];

    for (let i = 0; i < this.weights.length; i++) {
      const next: number[] = [];
      for (let j = 0; j < this.weights[i].length; j++) {
        let sum = this.biases[i][j];
        for (let k = 0; k < current.length; k++) {
          sum += current[k] * this.weights[i][j][k];
        }
        next.push(this.sigmoid(sum));
      }
      current = next;
      activations.push(current);
    }
    return activations;
  }

  train(inputs: number[], targets: number[]): number {
    const activations = this.feedForward(inputs);
    const deltas: number[][] = [];

    // 计算输出层误差
    const outputLayerIndex = this.layers.length - 1;
    const outputActivations = activations[outputLayerIndex];
    const outputDeltas: number[] = [];
    let loss = 0;
    for (let i = 0; i < outputActivations.length; i++) {
      const error = targets[i] - outputActivations[i];
      loss += error * error;
      outputDeltas.push(error * outputActivations[i] * (1 - outputActivations[i]));
    }
    deltas[outputLayerIndex - 1] = outputDeltas;

    // 反向传播误差
    for (let i = outputLayerIndex - 2; i >= 0; i--) {
      const currentDeltas: number[] = [];
      const nextDeltas = deltas[i + 1];
      const nextWeights = this.weights[i + 1];
      const currentActivations = activations[i + 1];

      for (let j = 0; j < this.layers[i + 1]; j++) {
        let error = 0;
        for (let k = 0; k < nextDeltas.length; k++) {
          error += nextDeltas[k] * nextWeights[k][j];
        }
        currentDeltas.push(error * currentActivations[j] * (1 - currentActivations[j]));
      }
      deltas[i] = currentDeltas;
    }

    // 更新权重和偏置
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          this.weights[i][j][k] += this.learningRate * deltas[i][j] * activations[i][k];
        }
        this.biases[i][j] += this.learningRate * deltas[i][j];
      }
    }

    return loss / 2;
  }
}
