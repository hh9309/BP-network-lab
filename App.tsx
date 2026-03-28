
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Legend, ReferenceLine, Label } from 'recharts';
import { BPNetwork } from './services/bpEngine';
import NetworkGraph from './components/NetworkGraph';
import { getAIInsights, ModelType } from './services/geminiService';

const App: React.FC = () => {
  const [layers, setLayers] = useState<number[]>([2, 4, 1]);
  const [learningRate, setLearningRate] = useState(0.5);
  const [sampleSize, setSampleSize] = useState(120);
  const [stopThreshold, setStopThreshold] = useState(0.005); 
  const [trainData, setTrainData] = useState<{ input: number[], output: number[] }[]>([]);
  const [testData, setTestData] = useState<{ input: number[], output: number[] }[]>([]);
  const [network, setNetwork] = useState<BPNetwork>(new BPNetwork([2, 4, 1], 0.5));
  const [isTraining, setIsTraining] = useState(false);
  const [history, setHistory] = useState<{ epoch: number; loss: number }[]>([]);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [testResults, setTestResults] = useState<any[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [aiInsight, setAiInsight] = useState<string>("");
  const [isAiLoading, setIsAiLoading] = useState(false);
  
  // AI 状态管理
  const [manualKey, setManualKey] = useState("");
  const [selectedModel, setSelectedModel] = useState<ModelType>('gemini_3_flash');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const trainingRef = useRef<any>(null);

  const generateData = useCallback(() => {
    const inputDim = layers[0];
    const outputDim = layers[layers.length - 1];
    const newData = Array.from({ length: sampleSize }, () => {
      const input = Array.from({ length: inputDim }, () => Math.random() * 2 - 1);
      const output = Array.from({ length: outputDim }, (_, i) => {
        const factor = i === 0 ? 1 : -0.7;
        const base = Math.sin(input[0] * Math.PI) * Math.cos((input[1] || 1) * Math.PI * factor);
        return Math.max(0.1, Math.min(0.9, (base + (i * 0.2) + 1) / 2)); 
      });
      return { input, output };
    });
    
    const splitIdx = Math.floor(newData.length * 0.8);
    setTrainData(newData.slice(0, splitIdx));
    setTestData(newData.slice(splitIdx));
    
    const newNet = new BPNetwork(layers, learningRate);
    setNetwork(newNet);
    setHistory([]);
    setEpoch(0);
    setCurrentLoss(0);
    setTestResults([]);
    setAiInsight("");
    setIsTraining(false);
  }, [layers, learningRate, sampleSize]);

  const runEvaluation = useCallback(() => {
    if (!network || testData.length === 0) return;
    setIsPredicting(true);
    
    setTimeout(() => {
      const outputDim = layers[layers.length - 1];
      const results = testData.map((d) => {
        const activations = network.feedForward(d.input);
        const predicted = activations[activations.length - 1];
        if (outputDim === 1) {
          return { x: d.output[0], y: predicted[0] };
        } else {
          return {
            actualX: d.output[0],
            actualY: d.output[1],
            predX: predicted[0],
            predY: predicted[1]
          };
        }
      });
      setTestResults(results);
      setIsPredicting(false);
    }, 300);
  }, [network, testData, layers]);

  useEffect(() => {
    generateData();
  }, [layers[0], layers[2], sampleSize, generateData]);

  const trainStep = useCallback(() => {
    if (!network || trainData.length === 0) return;
    let totalLoss = 0;
    trainData.forEach(point => {
      totalLoss += network.train(point.input, point.output);
    });
    const avgLoss = totalLoss / trainData.length;
    setCurrentLoss(avgLoss);
    setEpoch(prev => prev + 1);
    
    if (avgLoss <= stopThreshold) {
      setIsTraining(false);
      return;
    }

    if (epoch % 5 === 0) {
      setHistory(prev => [...prev.slice(-99), { epoch, loss: avgLoss }]);
    }
  }, [network, epoch, trainData, stopThreshold]);

  useEffect(() => {
    if (isTraining) {
      trainingRef.current = setInterval(trainStep, 20);
    } else {
      clearInterval(trainingRef.current);
    }
    return () => clearInterval(trainingRef.current);
  }, [isTraining, trainStep]);

  const handleGetInsight = async () => {
    if (!manualKey && !process.env.API_KEY) {
      setIsSettingsOpen(true);
      return;
    }
    setIsAiLoading(true);
    const result = await getAIInsights({ layers, learningRate }, { epoch, loss: currentLoss }, selectedModel, manualKey);
    
    if (result === "ERROR_KEY_NOT_FOUND") {
      setAiInsight("提示：API 密钥校验失败，请检查手工输入的密钥是否正确。");
      setIsSettingsOpen(true);
    } else {
      setAiInsight(result || "");
    }
    setIsAiLoading(false);
  };

  const getModelLabel = () => {
    switch(selectedModel) {
      case 'gemini_3_flash': return 'Gemini 3.0 Flash';
      case 'deepseek_r1': return 'DeepSeek R1 (Reasoner)';
      default: return '未知引擎';
    }
  };

  const outputDim = layers[layers.length - 1];

  return (
    <div className="min-h-screen p-4 md:p-8 bg-gray-50 flex flex-col gap-6 font-sans select-none text-slate-800">
      <header className="flex flex-col md:flex-row justify-between items-center bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
            <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path></svg>
          </div>
          <div>
            <h1 className="text-3xl font-extrabold text-blue-700 tracking-tight">NeuralLab</h1>
            <p className="text-slate-600 text-[10px] mt-0.5 font-bold uppercase tracking-[0.2em]">BP 神经网络实验室</p>
          </div>
        </div>
        <div className="flex gap-3 mt-4 md:mt-0">
          <button onClick={() => setIsTraining(!isTraining)} className={`px-8 py-2.5 rounded-full font-bold transition-all shadow-md flex items-center gap-2 ${isTraining ? 'bg-amber-500 text-white hover:bg-amber-600 scale-105' : 'bg-blue-600 text-white hover:bg-blue-700'}`}>
            {isTraining ? <><span className="w-2 h-2 bg-white rounded-full animate-ping"></span> 训练中</> : <><svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M4.5 3L15.5 10L4.5 17V3Z"/></svg> 启动训练</>}
          </button>
          <button onClick={generateData} className="px-6 py-2.5 rounded-full bg-white border border-slate-300 text-slate-800 font-bold hover:bg-slate-50 transition-all">刷新神经网络图</button>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* 左侧控制栏 */}
        <section className="lg:col-span-3 flex flex-col gap-6">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
            <h2 className="text-sm font-black mb-5 flex items-center gap-2 text-slate-900 uppercase tracking-widest"><span className="w-1.5 h-4 bg-blue-500 rounded-full"></span> 模型构建 (Neurons)</h2>
            <div className="space-y-6">
              {[
                { label: '输入层', key: 0, max: 5 },
                { label: '隐藏层', key: 1, max: 12 },
                { label: '输出层', key: 2, max: 2, min: 1 }
              ].map(layer => (
                <div key={layer.key}>
                  <div className="flex justify-between text-[10px] font-bold mb-1.5 uppercase tracking-wider text-slate-500"><span>{layer.label}</span><span className="text-blue-700 font-mono text-sm">{layers[layer.key]}</span></div>
                  <input type="range" min={layer.min || 1} max={layer.max} value={layers[layer.key]} onChange={e => {
                    const newLayers = [...layers];
                    newLayers[layer.key] = parseInt(e.target.value);
                    setLayers(newLayers);
                  }} className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" />
                </div>
              ))}
              <div className="pt-2">
                <div className="flex justify-between text-[10px] font-bold mb-1.5 uppercase tracking-wider text-slate-500"><span>学习率 (Alpha)</span><span className="text-blue-700 font-mono text-sm">{learningRate.toFixed(2)}</span></div>
                <input type="range" min="0.01" max="1" step="0.01" value={learningRate} onChange={e => setLearningRate(parseFloat(e.target.value))} className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" />
              </div>
              <div className="pt-2 border-t border-slate-100">
                <div className="flex justify-between text-[10px] font-bold mb-1.5 uppercase tracking-wider text-slate-500"><span>收敛阈值 (Stop at)</span><span className="text-red-600 font-mono text-sm">{stopThreshold.toFixed(2)}</span></div>
                <input type="range" min="0.0001" max="0.05" step="0.0005" value={stopThreshold} onChange={e => setStopThreshold(parseFloat(e.target.value))} className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-red-400" />
                <p className="text-[10px] text-slate-600 mt-2 italic font-medium">提示：当损失 (MSE) 低于此值时，训练自动停止。</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
            <h2 className="text-sm font-black mb-5 flex items-center gap-2 text-slate-900 uppercase tracking-widest"><span className="w-1.5 h-4 bg-emerald-500 rounded-full"></span> 样本生成 (Dataset)</h2>
            <div className="space-y-6">
              <div>
                <div className="flex justify-between text-[10px] font-bold mb-1.5 uppercase tracking-wider text-slate-500">
                  <span>样本总量 (Size)</span>
                  <span className="text-emerald-700 font-mono text-sm">{sampleSize}</span>
                </div>
                <input 
                  type="range" 
                  min="50" 
                  max="1000" 
                  step="10" 
                  value={sampleSize} 
                  onChange={e => setSampleSize(parseInt(e.target.value))} 
                  className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500" 
                />
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-100 space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] font-bold text-slate-500">训练集比例</span>
                  <span className="text-[10px] font-black text-slate-700">80% ({Math.floor(sampleSize * 0.8)})</span>
                </div>
                <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden flex">
                  <div className="h-full bg-emerald-500" style={{ width: '80%' }}></div>
                  <div className="h-full bg-blue-400" style={{ width: '20%' }}></div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[10px] font-bold text-slate-500">测试集比例</span>
                  <span className="text-[10px] font-black text-slate-700">20% ({sampleSize - Math.floor(sampleSize * 0.8)})</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white/40 border border-slate-100 p-6 rounded-3xl flex-grow flex flex-col justify-start">
            <h2 className="text-[11px] font-black mb-6 flex items-center gap-2 uppercase tracking-[0.2em] text-slate-600"><span className="w-4 h-[1px] bg-slate-300"></span> BP 算法</h2>
            <div className="space-y-6 text-[12px] leading-relaxed text-slate-700 font-medium">
              <section className="mb-2 px-1">
                <p className="text-slate-800 leading-relaxed font-bold border-b border-slate-100 pb-2">
                  误差反向传播 (Backpropagation) 是多层感知器训练的核心，其精髓在于利用链式法则逐层分配误差职责。
                </p>
              </section>
              <section className="border-l-2 border-blue-200 pl-4 py-1">
                <h3 className="font-bold text-slate-900 mb-2 italic">1. 前向传播 (Forward Pass)</h3>
                <p className="text-[10px] text-slate-500 mb-2">输入信号经各层神经元加权求和并激活，逐层产生映射结果：</p>
                <div className="font-mono bg-white/70 p-3 rounded-xl text-blue-800 border border-blue-100 shadow-inner">
                  z<sub>j</sub> = Σ w<sub>jk</sub> a<sub>k</sub> + b<sub>j</sub><br/>
                  a<sub>j</sub> = σ( z<sub>j</sub> )
                </div>
              </section>
              <section className="border-l-2 border-red-200 pl-4 py-1">
                <h3 className="font-bold text-slate-900 mb-2 italic">2. 反向传播 (Backward Pass)</h3>
                <p className="text-[10px] text-slate-500 mb-2">利用链式法则计算损失函数对每个权重的偏导数，逆向传递误差信号 δ：</p>
                <div className="space-y-2 font-mono bg-white/70 p-3 rounded-xl text-red-800 border border-red-100 shadow-inner">
                  <p>δ<sub>output</sub> = (a - y) ⊙ σ'(z)</p>
                  <p>δ<sub>hidden</sub> = (W<sup>T</sup>δ<sub>next</sub>) ⊙ σ'(z)</p>
                  <p className="text-[9px] text-red-400 mt-2">权重更新：Δw = -η · δ · a<sup>T</sup></p>
                </div>
              </section>
            </div>
          </div>
        </section>

        {/* 中间可视化栏 */}
        <section className="lg:col-span-6 flex flex-col gap-6">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex-grow min-h-[420px] flex flex-col relative overflow-hidden">
             <h2 className="text-sm font-black mb-4 flex items-center gap-2 text-slate-900 uppercase tracking-widest"><span className="w-1.5 h-4 bg-indigo-500 rounded-full"></span> 动态网络拓扑 (Dynamic Topology)</h2>
            <div className="flex-grow"><NetworkGraph layers={layers} weights={network.weights} /></div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-[300px] flex flex-col">
              <h2 className="text-xs font-black mb-4 flex items-center justify-between text-slate-800 uppercase tracking-wider">
                <div className="flex items-center gap-2"><span className="w-1.5 h-4 bg-red-400 rounded-full"></span> 损失收敛曲线</div>
                <div className="text-[10px] text-slate-600 font-mono font-bold">MSE: {currentLoss.toFixed(2)}</div>
              </h2>
              <div className="flex-grow">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" /><XAxis dataKey="epoch" hide /><YAxis domain={[0, 'auto']} tick={{fontSize: 9, fill: '#64748b', fontWeight: 'bold'}} axisLine={false} tickLine={false} /><Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: '10px'}} /><Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2.5} dot={false} isAnimationActive={false} /></LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-[300px] flex flex-col">
              <h2 className="text-xs font-black mb-4 flex items-center justify-between text-slate-800 uppercase tracking-wider">
                <div className="flex items-center gap-2"><span className="w-1.5 h-4 bg-emerald-400 rounded-full"></span> 模型预测评估</div>
                <button onClick={runEvaluation} disabled={isPredicting || isTraining} className="text-[10px] px-3 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100 transition-colors font-bold">执行预测</button>
              </h2>
              <div className="flex-grow relative">
                {currentLoss <= stopThreshold && epoch > 0 && (
                  <div className="absolute top-2 right-2 z-10 bg-emerald-100 text-emerald-800 text-[9px] px-3 py-1 rounded-full font-black border border-emerald-300 animate-bounce shadow-sm">
                    模型已收敛
                  </div>
                )}
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 25, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" />
                    {outputDim === 1 ? (
                      <>
                        <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{fontSize: 9, fontWeight: 'bold'}}>
                          <Label value="真实值" position="insideBottom" offset={-15} style={{fontSize: '10px', fontWeight: 'bold', fill: '#475569'}} />
                        </XAxis>
                        <YAxis type="number" dataKey="y" domain={[0, 1]} tick={{fontSize: 9, fontWeight: 'bold'}}>
                          <Label value="预测值" angle={-90} position="insideLeft" offset={10} style={{fontSize: '10px', fontWeight: 'bold', fill: '#475569'}} />
                        </YAxis>
                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#6366f1" strokeWidth={2} strokeDasharray="4 4" />
                        <Scatter name="预测点" data={testResults} fill="#3b82f6" fillOpacity={0.7} />
                      </>
                    ) : (
                      <>
                        <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{fontSize: 9, fontWeight: 'bold'}}>
                           <Label value="输出 1" position="insideBottom" offset={-15} style={{fontSize: '10px', fontWeight: 'bold', fill: '#475569'}} />
                        </XAxis>
                        <YAxis type="number" dataKey="y" domain={[0, 1]} tick={{fontSize: 9, fontWeight: 'bold'}}>
                           <Label value="输出 2" angle={-90} position="insideLeft" offset={10} style={{fontSize: '10px', fontWeight: 'bold', fill: '#475569'}} />
                        </YAxis>
                        <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{fontSize: '10px', fontWeight: 'bold', top: -10}} />
                        <Scatter name="真实目标" data={testResults.map(r => ({x: r.actualX, y: r.actualY}))} fill="#10b981" />
                        <Scatter name="预测结果" data={testResults.map(r => ({x: r.predX, y: r.predY}))} fill="#3b82f6" shape="cross" />
                        {testResults.map((r, i) => (
                           <ReferenceLine key={i} segment={[{ x: r.actualX, y: r.actualY }, { x: r.predX, y: r.predY }]} stroke="#ef4444" strokeWidth={1} strokeOpacity={0.4} />
                        ))}
                      </>
                    )}
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </section>

        {/* 右侧 AI 诊断栏 */}
        <section className="lg:col-span-3 flex flex-col relative">
          <div className="bg-white p-6 rounded-2xl shadow-xl shadow-slate-200/50 flex-grow flex flex-col border border-slate-100 overflow-hidden relative">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-indigo-600 rounded-lg shadow-md shadow-indigo-100">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                </div>
                <h2 className="text-lg font-black text-slate-900 tracking-tight">AI 模型诊断中心</h2>
              </div>
              <button 
                onClick={() => setIsSettingsOpen(true)} 
                className="p-2 text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 rounded-full transition-all"
                title="诊断设置"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
              </button>
            </div>

            <div className="flex-grow flex flex-col">
              <div className="flex-grow overflow-y-auto text-[12px] leading-relaxed text-slate-800 font-medium custom-scrollbar mb-5 pr-1">
                {aiInsight ? (
                  <div className="prose prose-sm prose-indigo max-w-none animate-in fade-in slide-in-from-bottom-2 duration-500">
                    <div dangerouslySetInnerHTML={{ __html: aiInsight.replace(/\n/g, '<br/>').replace(/### (.*?)<br\/>/g, '<h3 class="font-bold text-indigo-700 mt-4 mb-1 uppercase tracking-wider">$1</h3>') }} />
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center italic text-center py-20">
                    <svg className="w-12 h-12 mb-4 text-slate-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                    <p className="text-xs font-black text-slate-500">待研判诊断报告</p>
                    <p className="text-[11px] mt-1 text-slate-400 font-bold">请点击右上角齿轮<br/>手工录入 API 密钥后启动诊断</p>
                  </div>
                )}
              </div>

              <div className="mt-auto pt-4 border-t border-slate-100">
                <div className="flex items-center justify-between mb-3 px-1">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">诊断引擎：{manualKey ? getModelLabel() : '等待密钥授权'}</span>
                  {manualKey && <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse shadow-sm shadow-green-200"></div>}
                </div>
                <button 
                  onClick={handleGetInsight}
                  disabled={isAiLoading || epoch === 0}
                  className={`w-full py-4 rounded-2xl font-black text-sm text-white shadow-xl transition-all ${isAiLoading || epoch === 0 ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none' : 'bg-gradient-to-br from-indigo-600 to-blue-700 hover:scale-[1.02] active:scale-95 shadow-indigo-100'}`}
                >
                  {isAiLoading ? '专家研判中...' : epoch === 0 ? '需先启动训练' : '启动大模型诊断'}
                </button>
              </div>
            </div>

            {isSettingsOpen && (
              <div className="absolute inset-0 z-30 bg-white/98 backdrop-blur-md animate-in zoom-in-95 duration-200 p-6 flex flex-col">
                <div className="flex items-center justify-between mb-8">
                  <h3 className="text-sm font-black text-slate-900 uppercase tracking-widest flex items-center gap-2">
                    <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
                    诊断引擎配置
                  </h3>
                  <button onClick={() => setIsSettingsOpen(false)} className="p-2 text-slate-500 hover:text-slate-800 transition-colors">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                  </button>
                </div>

                <div className="space-y-6 flex-grow overflow-y-auto custom-scrollbar pr-1">
                  <div className="space-y-3">
                    <label className="text-[11px] font-bold text-slate-600 uppercase tracking-[0.2em] block mb-2">1. 输入 API-Key 密钥 (必填)</label>
                    <div className="relative group">
                      <input 
                        type="password" 
                        value={manualKey}
                        onChange={(e) => setManualKey(e.target.value)}
                        placeholder="粘贴您的 Gemini 或 DeepSeek API 密钥..."
                        className="w-full p-4 pr-12 rounded-2xl border-2 border-slate-200 bg-slate-50 text-xs font-mono font-bold focus:border-indigo-600 focus:bg-white focus:ring-4 focus:ring-indigo-50 outline-none transition-all placeholder:text-slate-400"
                      />
                      <div className={`absolute right-4 top-1/2 -translate-y-1/2 transition-opacity ${manualKey ? 'opacity-100' : 'opacity-0'}`}>
                        <div className="w-2.5 h-2.5 bg-green-500 rounded-full shadow-sm shadow-green-200"></div>
                      </div>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-1 italic">注：诊断功能完全依赖此密钥，系统不会持久化存储您的 Key。</p>
                  </div>
                  <div className={`space-y-4 transition-all ${!manualKey ? 'opacity-30 pointer-events-none grayscale' : 'opacity-100'}`}>
                    <label className="text-[11px] font-bold text-slate-600 uppercase tracking-[0.2em] block">2. 选定诊断大模型</label>
                    <div className="grid grid-cols-1 gap-3">
                      <button 
                        onClick={() => setSelectedModel('gemini_3_flash')}
                        className={`p-4 rounded-2xl border-2 text-left transition-all flex items-center justify-between ${selectedModel === 'gemini_3_flash' ? 'border-indigo-600 bg-indigo-50 shadow-sm' : 'border-slate-100 bg-white hover:border-slate-300'}`}
                      >
                        <div>
                          <p className="text-xs font-black text-slate-900 tracking-tight">Gemini 3.0 Flash</p>
                          <p className="text-[10px] text-slate-600 font-bold">Google 最新极速模型，推理精准</p>
                        </div>
                        {selectedModel === 'gemini_3_flash' && <div className="w-2.5 h-2.5 bg-indigo-600 rounded-full"></div>}
                      </button>

                      <button 
                        onClick={() => setSelectedModel('deepseek_r1')}
                        className={`p-4 rounded-2xl border-2 text-left transition-all flex items-center justify-between ${selectedModel === 'deepseek_r1' ? 'border-indigo-600 bg-indigo-50 shadow-sm' : 'border-slate-100 bg-white hover:border-slate-300'}`}
                      >
                        <div>
                          <p className="text-xs font-black text-slate-900 tracking-tight">DeepSeek R1 (Reasoner)</p>
                          <p className="text-[10px] text-slate-600 font-bold">国产最强推理模型，深度逻辑分析</p>
                        </div>
                        {selectedModel === 'deepseek_r1' && <div className="w-2.5 h-2.5 bg-indigo-600 rounded-full"></div>}
                      </button>
                    </div>
                  </div>

                  <div className="pt-2">
                    <label className="text-[11px] font-bold text-slate-600 uppercase tracking-[0.2em] block mb-4">3. 确认配置并返回实验室</label>
                    <button 
                      onClick={() => setIsSettingsOpen(false)}
                      disabled={!manualKey}
                      className="w-full py-4 rounded-2xl bg-slate-900 text-white text-sm font-black shadow-lg hover:bg-slate-800 transition-all disabled:opacity-20 active:scale-95"
                    >
                      确认配置并返回实验室
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* 核心概念科普切片模块 */}
      <section className="mt-4 animate-in fade-in slide-in-from-bottom-4 duration-700">
        <h2 className="text-xs font-black text-slate-400 uppercase tracking-[0.5em] mb-6 text-center">核心概念科普 · Core Concepts</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md transition-shadow flex items-start gap-4">
            <div className="w-10 h-10 bg-blue-50 rounded-lg flex-shrink-0 flex items-center justify-center text-blue-600 font-bold">01</div>
            <div>
              <h3 className="text-sm font-black text-slate-900 mb-2">神经元 (Neuron)</h3>
              <p className="text-[11px] leading-relaxed text-slate-600 font-medium">
                神经网络的基本处理单元。模仿生物神经元，它接收多路信号输入，通过<span className="text-blue-600">加权求和 (Σ)</span> 后，利用非线性<span className="text-blue-600">激活函数 (σ)</span> 决定是否向后传递信号。
              </p>
            </div>
          </div>
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md transition-shadow flex items-start gap-4">
            <div className="w-10 h-10 bg-indigo-50 rounded-lg flex-shrink-0 flex items-center justify-center text-indigo-600 font-bold">02</div>
            <div>
              <h3 className="text-sm font-black text-slate-900 mb-2">神经网络 (Neural Network)</h3>
              <p className="text-[11px] leading-relaxed text-slate-600 font-medium">
                由大量神经元相互连接形成的计算网络。分为<span className="text-indigo-600">输入层</span>、<span className="text-indigo-600">隐藏层</span>和<span className="text-indigo-600">输出层</span>。这种多层级结构赋予了模型处理复杂非线性映射的能力。
              </p>
            </div>
          </div>
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md transition-shadow flex items-start gap-4">
            <div className="w-10 h-10 bg-emerald-50 rounded-lg flex-shrink-0 flex items-center justify-center text-emerald-600 font-bold">03</div>
            <div>
              <h3 className="text-sm font-black text-slate-900 mb-2">BP 神经网络 (BP Neural Network)</h3>
              <p className="text-[11px] leading-relaxed text-slate-600 font-medium">
                采用误差反向传播算法的多层前向网络。它通过<span className="text-emerald-600">信号正向传播</span>计算输出结果，并利用<span className="text-emerald-600">误差反向传播</span>通过梯度下降不断修正连接权重，是目前最成熟的神经网络模型之一。
              </p>
            </div>
          </div>
        </div>
      </section>
      
      <footer className="mt-2 text-center text-[10px] text-slate-500 pb-4 uppercase tracking-[0.4em] font-bold">
        NeuralLab &copy; 2024 · INTELLIGENCE EMPOWERED BY GOOGLE GENAI
      </footer>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #cbd5e1; }
        
        input[type=range]::-webkit-slider-thumb {
          -webkit-appearance: none;
          height: 14px;
          width: 14px;
          border-radius: 50%;
          background: #2563eb;
          cursor: pointer;
          border: 2.5px solid white;
          box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        }
        
        input[type=range].accent-emerald-500::-webkit-slider-thumb {
          background: #10b981;
        }

        input[type=range].accent-red-400::-webkit-slider-thumb {
          background: #ef4444;
        }
      `}</style>
    </div>
  );
};

export default App;
