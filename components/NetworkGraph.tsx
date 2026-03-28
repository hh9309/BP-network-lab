
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface NetworkGraphProps {
  layers: number[];
  weights: number[][][];
}

const NetworkGraph: React.FC<NetworkGraphProps> = ({ layers, weights }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    // 增加顶部边距以容纳标签并创建“空行”感
    const marginTop = 100; 
    const marginBottom = 80;
    const marginSide = 80;

    // 获取权重最大绝对值用于归一化
    let maxAbsWeight = 0;
    weights.forEach(layer => layer.forEach(neuron => neuron.forEach(w => {
      const absW = Math.abs(w);
      if (absW > maxAbsWeight) maxAbsWeight = absW;
    })));
    if (maxAbsWeight === 0) maxAbsWeight = 1;

    // 定义滤镜和箭头
    const defs = svg.append("defs");
    
    // 1. 节点外部阴影
    const filter = defs.append("filter")
      .attr("id", "drop-shadow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
    filter.append("feDropShadow")
      .attr("dx", "0")
      .attr("dy", "2")
      .attr("stdDeviation", "3")
      .attr("flood-color", "#000000")
      .attr("flood-opacity", "0.15");

    // 2. 节点内部渐变 (根据层级)
    const createGradient = (id: string, color1: string, color2: string) => {
      const grad = defs.append("linearGradient")
        .attr("id", id)
        .attr("x1", "0%").attr("y1", "0%")
        .attr("x2", "0%").attr("y2", "100%");
      grad.append("stop").attr("offset", "0%").attr("stop-color", color1);
      grad.append("stop").attr("offset", "100%").attr("stop-color", color2);
    };
    createGradient("grad-input", "#eff6ff", "#dbeafe");
    createGradient("grad-hidden", "#f5f3ff", "#ede9fe");
    createGradient("grad-output", "#ecfdf5", "#d1fae5");

    // 3. 有向箭头
    defs.append("marker")
      .attr("id", "arrow-head")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 32)
      .attr("refY", 0)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#1e293b");

    const layerSpacing = (width - 2 * marginSide) / (layers.length - 1 || 1);
    
    const nodes: any[] = [];
    const nodeMap: { [key: string]: any } = {};

    // 绘制层级标题 - 将 y 坐标下移至 65，并在顶部增加间距
    layers.forEach((_, lIdx) => {
      const x = marginSide + lIdx * layerSpacing;
      let label = "隐藏层 (Hidden)";
      let colorClass = "fill-indigo-950";
      if (lIdx === 0) {
        label = "输入层 (Input)";
        colorClass = "fill-blue-950";
      } else if (lIdx === layers.length - 1) {
        label = "输出层 (Output)";
        colorClass = "fill-emerald-950";
      }

      svg.append("text")
        .attr("x", x)
        .attr("y", 65) // 下移标题位置，与顶部标签拉开距离
        .attr("text-anchor", "middle")
        .attr("font-size", "11px")
        .attr("font-weight", "900")
        .attr("class", `${colorClass} uppercase tracking-[0.3em]`)
        .text(label);
    });

    // 创建节点坐标
    layers.forEach((count, lIdx) => {
      const x = marginSide + lIdx * layerSpacing;
      const vertSpacing = (height - marginTop - marginBottom) / (count || 1);
      const startY = marginTop + (height - marginTop - marginBottom - (count - 1) * vertSpacing) / 2;

      for (let i = 0; i < count; i++) {
        const id = `l${lIdx}n${i}`;
        const node = { id, x, y: startY + i * vertSpacing, layer: lIdx, index: i + 1 };
        nodes.push(node);
        nodeMap[id] = node;
      }
    });

    // 绘制权重连接线
    const edges: any[] = [];
    for (let l = 0; l < weights.length; l++) {
      for (let j = 0; j < weights[l].length; j++) {
        for (let k = 0; k < weights[l][j].length; k++) {
          if (nodeMap[`l${l}n${k}`] && nodeMap[`l${l+1}n${j}`]) {
            edges.push({
              source: nodeMap[`l${l}n${k}`],
              target: nodeMap[`l${l+1}n${j}`],
              weight: weights[l][j][k]
            });
          }
        }
      }
    }

    const lineGroup = svg.append("g")
      .selectAll("line")
      .data(edges)
      .enter()
      .append("line")
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y)
      .attr("stroke", d => d.weight > 0 ? "#1e40af" : "#b91c1c")
      .attr("stroke-width", d => (Math.abs(d.weight) / maxAbsWeight) * 4 + 0.6)
      .attr("opacity", d => (Math.abs(d.weight) / maxAbsWeight) * 0.7 + 0.2)
      .attr("marker-end", "url(#arrow-head)");
    
    // 为连接线添加原生 Tooltip 提示
    lineGroup.append("title")
      .text(d => `权重 (Weight): ${d.weight.toFixed(2)}`);

    // 绘制高级神经元模型 (Σ | σ)
    const nodeGroups = svg.append("g")
      .selectAll("g")
      .data(nodes)
      .enter()
      .append("g")
      .attr("filter", "url(#drop-shadow)");

    const radius = 22;
    nodeGroups.append("circle")
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", radius)
      .attr("fill", d => {
        if (d.layer === 0) return "url(#grad-input)";
        if (d.layer === layers.length - 1) return "url(#grad-output)";
        return "url(#grad-hidden)";
      })
      .attr("stroke", d => {
        if (d.layer === 0) return "#1e40af";
        if (d.layer === layers.length - 1) return "#065f46";
        return "#3730a3";
      })
      .attr("stroke-width", 2);

    nodeGroups.append("line")
      .attr("x1", d => d.x).attr("y1", d => d.y - radius + 5)
      .attr("x2", d => d.x).attr("y2", d => d.y + radius - 5)
      .attr("stroke", d => {
        if (d.layer === 0) return "#3b82f6";
        if (d.layer === layers.length - 1) return "#10b981";
        return "#6366f1";
      })
      .attr("stroke-width", 1.2)
      .attr("stroke-dasharray", "2 1");

    nodeGroups.append("text")
      .attr("x", d => d.x - 8)
      .attr("y", d => d.y + 4)
      .attr("text-anchor", "middle")
      .attr("font-size", "11px")
      .attr("font-family", "serif")
      .attr("font-weight", "900")
      .attr("fill", d => d.layer === 0 ? "#0f172a" : d.layer === layers.length - 1 ? "#064e3b" : "#1e1b4b")
      .text("Σ");

    nodeGroups.append("text")
      .attr("x", d => d.x + 8)
      .attr("y", d => d.y + 4)
      .attr("text-anchor", "middle")
      .attr("font-size", "11px")
      .attr("font-family", "serif")
      .attr("font-weight", "900")
      .attr("fill", d => d.layer === 0 ? "#0f172a" : d.layer === layers.length - 1 ? "#064e3b" : "#1e1b4b")
      .text("σ");

    const badge = nodeGroups.append("g")
      .attr("transform", d => `translate(${d.x}, ${d.y - radius - 2})`);
    
    badge.append("rect")
      .attr("x", -10).attr("y", -8)
      .attr("width", 20).attr("height", 10)
      .attr("rx", 4)
      .attr("fill", d => {
        if (d.layer === 0) return "#1e3a8a";
        if (d.layer === layers.length - 1) return "#064e3b";
        return "#312e81";
      });

    badge.append("text")
      .attr("text-anchor", "middle")
      .attr("y", -1)
      .attr("font-size", "7.5px")
      .attr("font-weight", "black")
      .attr("fill", "white")
      .text(d => `N${d.index}`);

    // 图例说明
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 160}, ${height - 110})`);

    legend.append("rect")
      .attr("width", 145).attr("height", 95)
      .attr("rx", 12)
      .attr("fill", "white")
      .attr("fill-opacity", 0.95)
      .attr("stroke", "#94a3b8")
      .attr("stroke-width", 1.5);

    legend.append("text")
      .attr("x", 12).attr("y", 22)
      .attr("font-size", "10px")
      .attr("font-weight", "black")
      .attr("fill", "#020617")
      .attr("class", "uppercase tracking-widest")
      .text("神经网络图例");

    legend.append("line")
      .attr("x1", 12).attr("y1", 38).attr("x2", 42).attr("y2", 38)
      .attr("stroke", "#1e40af").attr("stroke-width", 3.5);
    legend.append("text").attr("x", 48).attr("y", 41).attr("font-size", "9px").attr("font-weight", "900").attr("fill", "#1e293b").text("正权重 (兴奋)");

    legend.append("line")
      .attr("x1", 12).attr("y1", 53).attr("x2", 42).attr("y2", 53)
      .attr("stroke", "#991b1b").attr("stroke-width", 3.5);
    legend.append("text").attr("x", 48).attr("y", 56).attr("font-size", "9px").attr("font-weight", "900").attr("fill", "#1e293b").text("负权重 (抑制)");

    legend.append("circle").attr("cx", 27).attr("cy", 75).attr("r", 10).attr("fill", "#f8fafc").attr("stroke", "#1e293b").attr("stroke-width", 1.5);
    legend.append("text").attr("x", 48).attr("y", 78).attr("font-size", "9px").attr("font-weight", "900").attr("fill", "#1e293b").text("Σ 求和 | σ 激活");

  }, [layers, weights]);

  return (
    <div className="w-full h-full bg-white rounded-xl shadow-inner border border-blue-100 overflow-hidden relative">
      <div className="absolute top-4 left-4 flex items-center gap-2">
         <div className="w-2.5 h-2.5 bg-blue-700 rounded-full animate-pulse shadow-sm shadow-blue-200"></div>
         <span className="text-[11px] text-slate-950 font-black uppercase tracking-widest pointer-events-none">实时物理神经元模拟</span>
      </div>
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};

export default NetworkGraph;
