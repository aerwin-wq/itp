const NAT_W = 150;
const NAT_H = 150;

function setup() {
  createCanvas(500, 500);
  background(255);
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
  // Test calls
  drawObject(0, 0, 1);
  drawObject(220, 50, 1.2);
}
