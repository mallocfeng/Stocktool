import * as echarts from 'echarts';

let morandiRegistered = false;

const registerMorandiTheme = () => {
  if (morandiRegistered) return;
  echarts.registerTheme('morandi', {
    color: ['#8d94c2', '#d38a9e', '#86a89f', '#e3c08d', '#c6a3c7', '#86b9c8'],
    backgroundColor: '#f6f2ed',
    title: {
      textStyle: { color: '#5a5550' },
    },
    textStyle: {
      color: '#5a5550',
    },
    line: {
      smooth: true,
    },
    categoryAxis: {
      axisLine: { lineStyle: { color: '#b1aaa2' } },
      axisLabel: { color: '#5a5550' },
      splitLine: { lineStyle: { color: '#e4dbd1' } },
    },
    valueAxis: {
      axisLine: { lineStyle: { color: '#b1aaa2' } },
      axisLabel: { color: '#5a5550' },
      splitLine: { lineStyle: { color: '#e4dbd1' } },
    },
    dataZoom: {
      fillerColor: 'rgba(156, 143, 184, 0.2)',
      backgroundColor: 'rgba(156, 143, 184, 0.05)',
      handleStyle: {
        color: '#9c8fb8',
      },
    },
    legend: {
      textStyle: { color: '#5a5550' },
    },
  });
  morandiRegistered = true;
};

export const resolveEchartTheme = (theme) => {
  if (theme === 'dark') return 'dark';
  if (theme === 'morandi') {
    registerMorandiTheme();
    return 'morandi';
  }
  return undefined;
};
