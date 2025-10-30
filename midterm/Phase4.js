
const NAT_W = 150;
const NAT_H = 150;

let COLS = 10;
let ROWS = 10;

function setup() {
  createCanvas(600, 600);
  noLoop();
}

function drawObject(x, y, s) {
  push();
  translate(x, y);
  scale(s);
  noFill();
  stroke(0);

  strokeWeight(6);
  ellipse(75, 95, 80, 64);

  strokeWeight(5);
  line(55, 88, 95, 104);
  line(60, 105, 90, 85);

  strokeWeight(10);
  line(115, 25, 115, 120);

  strokeWeight(6);
  line(112, 115, 120, 108);
  line(37, 118, 22, 130);

  pop();
}

function draw() {
  background(255);
  const cellW = width / COLS;
  const cellH = height / ROWS;

  const baseS = 0.9 * Math.min(cellW / NAT_W, cellH / NAT_H);

  for (let cx = 0; cx < COLS; cx++) {
    for (let cy = 0; cy < ROWS; cy++) {

      const ox = cx * cellW;
      const oy = cy * cellH;


      const x = ox + (cellW - NAT_W * baseS) / 2;
      const y = oy + (cellH - NAT_H * baseS) / 2;

      drawObject(x, y, baseS);
    }
  }
}
