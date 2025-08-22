import WebSocket from 'ws';

const ws = new WebSocket('ws://localhost:3001');
ws.on('open', () => {
  console.log('connected');
  ws.send('USER|Suggest a sofa and rug size for a 4x5 m room. Budget ₾1500.');
});
ws.on('message', (data) => console.log(data.toString()));
ws.on('close', () => console.log('closed'));
ws.on('error', (e) => console.error('ws error:', e.message));
