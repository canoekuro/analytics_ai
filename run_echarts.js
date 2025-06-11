const { createCanvas } = require('canvas');
const echarts = require('echarts');

(async () => {
  const optionStr = process.argv[2] || '{}';
  const canvas = createCanvas(800, 600);
  const chart = echarts.init(canvas);
  chart.setOption(JSON.parse(optionStr));
  await new Promise((r) => chart.on('finished', r));
  const buf = canvas.toBuffer('image/png');
  console.log(buf.toString('base64'));
})();
