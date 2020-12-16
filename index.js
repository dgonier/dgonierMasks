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

const overlay = document.getElementById('overlay-container')
const mainApp = document.getElementById('main-app')

const loadModel = async () => {
  const model = await tf.loadLayersModel('./model.json');
  const webcamElement = document.getElementById('video');
  const webcam = await tf.data.webcam(webcamElement);
// model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

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

    if (probs[0] > 0.90) {
      noMask = false
      overlay.style.display = "none"
      mainApp.style.display = "block"
    }
    // console.log(result)
    // console.log(result.dataSync())
    let maskWearProb = Math.round(probs[0]*100)
    
    document.getElementById('console').innerText = `
      ${maskWearProb}% prediction of wearing mask.
    `

    // document.getElementById('console').innerText = `
    //   prediction: ${result[0].className}\n
    //   probability: ${result[0].probability}
    // `;
    // status(`${result[0].className}`)
    // Dispose the tensor to release the memory.
    // img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await sleep(1000);
    // await tf.nextFrame();
  }
}

loadModel()




// todo note change to setInterval
// window.setTimeout(function(){
//   console.log('hi')
//   var context = canvas.getContext("2d").drawImage(video, 0, 0, 640, 480);
//   let input = canvas.toDataURL("image/jpg")
//   let img = document.getElementById('cat');
//   img.src = input;
//   demo(img)
// }, 3000)