let xs = [];
let ys = [];

let a, b, c, d;

let addingPoints = false;
let lrChanged = false;

let learningRate = 0.1;
let optimizer = tf.train.sgd(learningRate);

const functionDegree = document.getElementById('function-degree');
const learningRateInfo = document.getElementById('lr-info');
const learningRateValue = document.getElementById('lr-value');

function setup() {
    createCanvas(400, 400);

    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    d = tf.variable(tf.scalar(random(-1, 1)));

    learningRateInfo.innerHTML = learningRate;
    learningRateValue.value = learningRate;

    learningRateValue.addEventListener('change', () => {
        learningRate = learningRateValue.value;
        learningRateInfo.innerHTML = learningRate;
        lrChanged = true;
    });

}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function predict(xs, fDegree = 3) {
    const tfxs = tf.tensor1d(xs);
    let ys;
    // y = ax + b
    if (fDegree == 1) {
        ys = tfxs.mul(a).add(b);
    }
    // y = ax^2 + bx + c
    if (fDegree == 2) {
        ys = tfxs.square().mul(a).add(tfxs.mul(b)).add(c);
    }
    // y = ax^3 + bx^2 + cx + d
    if (fDegree == 3) {
        ys = tfxs.pow(tf.scalar(3)).mul(a).add(tfxs.square().mul(b)).add(tfxs.mul(c)).add(d);
    }
    return ys;
}

function addPoints() {
    // normalize coordinates (values between 0-1)
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);

    xs.push(x);
    ys.push(y);
}

function mousePressed() {
    if (mouseX < width && mouseX > 0 && mouseY < height && mouseY > 0) {
        addingPoints = true;
    }
}

function mouseReleased() {
    addingPoints = false;
}

function draw() {
    background(255);
    stroke(0);
    strokeWeight(6);

    let fDegree = functionDegree.options[functionDegree.selectedIndex].value

    if (addingPoints) {
        addPoints();
    } else {
        tf.tidy(() => {
            if (xs.length > 0) {
                const tfys = tf.tensor1d(ys);
                if (lrChanged) {
                    optimizer.setLearningRate(learningRate);
                    lrChanged = false;
                }
                optimizer.minimize(() => loss(predict(xs, fDegree), tfys));
            }
        });
    }

    for (let i = 0; i < xs.length; i++) {
        // reverse normalization (values between 0 - height and 0 - width)
        let px = map(xs[i], -1, 1, 0, width);
        let py = map(ys[i], -1, 1, height, 0);
        // draw a point
        point(px, py);
    }

    // draw the line
    
    // const xvals = [-1, 1];
    const curveX = [];
    for (let x = -1; x < 1.01; x+= 0.05) {
        curveX.push(x);
    }
    const yvals = tf.tidy(() => predict(curveX, fDegree));
    
    let curveY = yvals.dataSync();
    yvals.dispose();

    // draw line
    beginShape();
    noFill();
    stroke(0);
    strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width);
        let y = map(curveY[i], -1, 1, height, 0);
        vertex(x, y);
    }
    endShape();

    // console.log(tf.memory().numTensors);

}