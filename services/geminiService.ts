
import { GoogleGenAI } from "@google/genai";

export type ModelType = 'gemini_3_flash' | 'deepseek_r1';

export const getAIInsights = async (config: any, stats: any, modelType: ModelType = 'gemini_3_flash', manualApiKey?: string) => {
  const apiKey = manualApiKey;
  
  if (!apiKey) {
    return "ERROR_KEY_NOT_FOUND";
  }

  const prompt = `
    你是一名深度学习专家。请针对以下 BP 神经网络实验室的当前状态提供简明、专业的诊断报告。
    
    网络配置：
    - 层级结构 (Neurons per layer)：${config.layers.join(' -> ')}
    - 学习率 (Learning Rate)：${config.learningRate}
    
    训练数据：
    - 已执行轮次 (Epochs)：${stats.epoch}
    - 当前均方误差 (Current Loss)：${stats.loss.toFixed(6)}
    
    诊断要求：
    1. 评估当前网络拓扑结构与任务复杂度的匹配度。
    2. 分析损失收敛趋势（如：学习率是否过大导致震荡，或过小导致陷入局部最优）。
    3. 给出具体的超参数调整建议。
    
    请使用标准中文 Markdown 格式回答，保持在 200-300 字左右，语气优雅专业。
  `;

  if (modelType === 'gemini_3_flash') {
    // 使用 Google Gemini SDK
    const ai = new GoogleGenAI({ apiKey });
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: prompt,
      });
      return response.text;
    } catch (error: any) {
      console.error("Gemini Error:", error);
      if (error?.message?.includes("API_KEY_INVALID") || error?.message?.includes("not authorized")) {
        return "ERROR_KEY_NOT_FOUND";
      }
      return `Gemini 诊断模块由于连接原因未能响应，请检查密钥权限或模型可用性。`;
    }
  } else if (modelType === 'deepseek_r1') {
    // 使用 DeepSeek API (OpenAI 兼容)
    try {
      const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'deepseek-reasoner', // DeepSeek R1 的模型 ID
          messages: [
            { role: 'system', content: '你是一名深度学习专家。' },
            { role: 'user', content: prompt }
          ],
          stream: false
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        if (response.status === 401) return "ERROR_KEY_NOT_FOUND";
        throw new Error(errData.error?.message || 'DeepSeek API Error');
      }

      const data = await response.json();
      return data.choices[0].message.content;
    } catch (error: any) {
      console.error("DeepSeek Error:", error);
      return `DeepSeek 诊断模块未能响应：${error.message}。请确保您使用的是有效的 DeepSeek API Key。`;
    }
  }

  return "未知的模型类型";
};
