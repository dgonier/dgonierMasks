let CLASSES = {
  0: 'Wearing Mask',
  1: 'Not Wearing Mask',
};


const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captures = []

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
    video.srcObject = stream
    video.play();
  });

}

const IMAGE_SIZE = 192;

const loadModel = async () => {
  const model = await tf.loadLayersModel('./model.json');
  const webcamElement = document.getElementById('video');
  let noMask = true
  const sleep = m => new Promise(r => setTimeout(r, m))


  while (noMask) {
    var context = canvas.getContext("2d").drawImage(webcamElement, 0, 0, 640, 480);
    let input = canvas.toDataURL("image/jpg")
    let img = document.getElementById('imageInput');
    img.width = IMAGE_SIZE;
    img.height = IMAGE_SIZE;
    img.src = input;

    const imgPixels = tf.browser.fromPixels(webcamElement).toFloat();
    // const imgPixels = tf.fromPixels(webcamElement).toFloat();
    const normalized = imgPixels.div(255.0);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    // console.log(batched.dataSync())
    const result = await model.predict(batched);
    const probs = result.dataSync()

    if (probs[0] > 0.95) {
      noMask = false
    }

    document.getElementById('console').innerText = `
      ${probs[0]}% of wearing mask
    `

    await sleep(1000);
    // await tf.nextFrame();
  }
}

loadModel()
