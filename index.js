let mobilenet;
let model;
const dataset = new RPSDataset();
var bharataSamples=0, kathkaliSamples=0, manipuriSamples=0;
let isPredicting = false;
var cat_id=0;
var im=0,jm=0,km=0,lm=0;
async function loadMobilenet() {

    // const mobilenet = await tf.loadLayersModel("model.json");
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  // console.log(mobilenet);
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  // console.log(layer);
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  // console.log( mobilenet.outputs[0]);
  // return tf.model({inputs: mobilenet.inputs, outputs: mobilenet.output});
}

var ik = 0;
var width = 0;
function move() {
    var elem = document.getElementById("myBar");    
      if (width >= 100) {
        // clearInterval(id);
        ik = 0;
      } else {
        width=width+10;
        elem.style.width = width + "%";
        elem.innerHTML = width  + "%";
  }
}
//400 times training
async function train() {

  dataset.ys = null;
  dataset.encodeLabels(2);
  //added own layer in pretrained mobilemnet
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 200, activation: 'relu'}),
      tf.layers.dense({ units: 2, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  console.log(dataset.xs,dataset.ys);
  model.fit(dataset.xs, dataset.ys, {
    epochs: 100,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        move();
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			bharataSamples++;
			document.getElementById("bharatsamples").innerText = "BharataNatyam samples:" + bharataSamples;
            document.getElementById('bharat_enter').innerHTML = "Enter BharataNatyam pics";
			bharata(0);
			break;
		case "1":
			kathkaliSamples++;
			document.getElementById("kathkalisamples").innerText = "Kathakkali samples:" + kathkaliSamples;
	        document.getElementById('kathkali_enter').innerHTML = "Enter Kathkali pics";
			kathkali(1);
			break;
		case "2":
			manipuriSamples++;
			document.getElementById("manipurisamples").innerText = "Manipuri samples:" + manipuriSamples;
		    document.getElementById('manipuri_enter').innerHTML = "Enter Manipuri pics";
			manipuri(2);
			break;
	}

}

function bharata(label){
    
    cat_id = label ;
    if(window.File && window.FileList && window.FileReader)
    {
        var filesInput = document.getElementById('files');
        filesInput.addEventListener('change', function(event){
            var files = event.target.files; //FileList object
            var output = document.getElementById('result');
            for(var i = 0; i< files.length; i++)
            {
                var file = files[i];
                //Only pics
                if(!file.type.match('image'))
                    continue;
                var picReader = new FileReader();
                picReader.addEventListener('load',function(event){
                    var picFile = event.target;
                    var div = document.createElement('span');//span so that image can be inline
                    im=im+1;
                    var ids = "im"+ im.toString();
                    var cats = cat_id.toString();

                    div.innerHTML = "<img class='thumbnail' style = 'width:224px ;height:224px;' id = '" +cats  + ids  + "' src='" + picFile.result + "'" +  "title='" + picFile.name + "'/>";
  
                    output.insertBefore(div,null);

                    const webcams = document.getElementById(cats + "im"+ im.toString());
   
                    const webcamImages = tf.browser.fromPixels(webcams);
       
                    const reversedImages = webcamImages.reverse(1);

                    const size = Math.min(reversedImages.shape[0], reversedImages.shape[1]);
				    const centerHeight = reversedImages.shape[0] / 2;
				    const beginHeight = centerHeight - (size / 2);
				    const centerWidth = reversedImages.shape[1] / 2;
				 
				    const beginWidth = centerWidth - (size / 2);
				    const croppedImage= reversedImages.slice([beginHeight, beginWidth, 0], [size, size, 3]);
				    const batchedImage = croppedImage.expandDims(0);
				   
				    const final_img = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
				    dataset.addExample(mobilenet.predict(final_img), label);
				
                });
                //Read the image
                picReader.readAsDataURL(file);
            }
        });
 	}

    else
    {
        console.log('Your browser does not support File API');
    }
}




function kathkali(label){

    cat_id = label ;

    if(window.File && window.FileList && window.FileReader)
    {
        var filesInput = document.getElementById('filess');
        filesInput.addEventListener('change', function(event){
            var files = event.target.files; //FileList object
            var output = document.getElementById('results');
            for(var i = 0; i< files.length; i++)
            {
                var file = files[i];

                if(!file.type.match('image'))
                    continue;
                var picReader = new FileReader();
                picReader.addEventListener('load',function(event){
                    var picFile = event.target;
                    var div = document.createElement('span');
                    jm=jm+1;
                    var ids = "im"+ jm.toString();
                    var cats = cat_id.toString();
                    div.innerHTML = "<img class='thumbnail' style = 'width:224px ;height:224px;' id = '" +cats  + ids  + "' src='" + picFile.result + "'" +  "title='" + picFile.name + "'/>";
        
                    output.insertBefore(div,null);

                    const webcams = document.getElementById(cats + "im"+ jm.toString());

                    const webcamImages = tf.browser.fromPixels(webcams);
              
                    const reversedImages = webcamImages.reverse(1);

                    const size = Math.min(reversedImages.shape[0], reversedImages.shape[1]);
				    const centerHeight = reversedImages.shape[0] / 2;
				    const beginHeight = centerHeight - (size / 2);
				    const centerWidth = reversedImages.shape[1] / 2;
				   
				    const beginWidth = centerWidth - (size / 2);
				    const croppedImage= reversedImages.slice([beginHeight, beginWidth, 0], [size, size, 3]);
				    const batchedImage = croppedImage.expandDims(0);
			
				    const final_img = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
				    dataset.addExample(mobilenet.predict(final_img), label);
	
                });
                //Read the image
                picReader.readAsDataURL(file);
            }
        });
 	}

    else
    {
        console.log('Your browser does not support File API');
    }
}






 function manipuri(label){

    var cat_id = label ;

    if(window.File && window.FileList && window.FileReader)
    {
        var filesInput = document.getElementById('filesss');
        filesInput.addEventListener('change', function(event){
            var files = event.target.files; //FileList object
            var output = document.getElementById('resultss');
            for(var i = 0; i< files.length; i++)
            {
                var file = files[i];
                //Only pics
                if(!file.type.match('image'))
                    continue;
                var picReader = new FileReader();
                picReader.addEventListener('load',function(event){
                    var picFile = event.target;
                    console.log(i);

                    var div = document.createElement('span');
                    km = km+1;
                    var ids = "im"+ km.toString();
                    var cats = cat_id.toString();

              
                    div.innerHTML = "<img class='thumbnail' style = 'width:224px ;height:224px;' id = '" +cats  + ids  + "' src='" + picFile.result + "'" +  "title='" + picFile.name + "'/>";
               
                    output.insertBefore(div,null);

                    const webcams = document.getElementById(cats + "im"+ km.toString());
                  
                    const webcamImages = tf.browser.fromPixels(webcams);

                    const reversedImages = webcamImages.reverse(1);

                    const size = Math.min(reversedImages.shape[0], reversedImages.shape[1]);
				    const centerHeight = reversedImages.shape[0] / 2;
				    const beginHeight = centerHeight - (size / 2);
				    const centerWidth = reversedImages.shape[1] / 2;
			
				    const beginWidth = centerWidth - (size / 2);
				    const croppedImage= reversedImages.slice([beginHeight, beginWidth, 0], [size, size, 3]);
				    const batchedImage = croppedImage.expandDims(0);
				
				    const final_img = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
				    dataset.addExample(mobilenet.predict(final_img), label);
	
                });
                //Read the image
                picReader.readAsDataURL(file);
            }
        });
 	}

    else
    {
        console.log('Your browser does not support File API');
    }
}



async function predict(img) {
  // while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      // const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    console.log("ddddddd");
    switch(classId){
		case 0:
			predictionText = "I see BharataNatyam";
			break;
		case 1:
			predictionText = "I see Kathakkali";
			break;
		case 2:
			predictionText = "I see Manipuri";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
    predictedClass.dispose();
  
  }
var loadFile = function(event) {
    var reader = new FileReader();
    reader.onload = function(){
      var output = document.getElementById('output');
      output.src = reader.result;
      const webcams = document.getElementById("output");

                    const webcamImages = tf.browser.fromPixels(webcams);

                    const reversedImages = webcamImages.reverse(1);

                    const size = Math.min(reversedImages.shape[0], reversedImages.shape[1]);
                    const centerHeight = reversedImages.shape[0] / 2;
                    const beginHeight = centerHeight - (size / 2);
                    const centerWidth = reversedImages.shape[1] / 2;

                    const beginWidth = centerWidth - (size / 2);
                    const croppedImage= reversedImages.slice([beginHeight, beginWidth, 0], [size, size, 3]);
                    const batchedImage = croppedImage.expandDims(0);

                    const final_img = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
                    predict(final_img);
    };
    reader.readAsDataURL(event.target.files[0]);
  };


//  function handlepre(){

//      document.getElementById('test_enter').innerHTML = "Enter testing images"; 
//     if(window.File && window.FileList && window.FileReader)
//     {
//         var filesInput = document.getElementById('filessss');
//         // console.log(filesInput);
//         filesInput.addEventListener('change', function(event){
//             var op = document.getElementById('resultsss');
//             op.innerHTML="";
//             files = event.target.files; //FileList object
            
//             for( var i = files.length-1; i>=0; i--)
//             {
//                 var file = files[i];
//                 //Only pics
//                 if(!file.type.match('image'))
//                     continue;
//                 var picReader = new FileReader();
//                 picReader.addEventListener('load',function(event){

//                     var picFile = event.target;
//                     div = document.createElement('span');

//                     var ids = "im"+ lm.toString();
//                     var output = document.getElementById('resultsss');
//                     div.innerHTML = "<img class='thumbnail' style = 'width:224px ;height:224px;' id = '" + ids  + "' src='" + picFile.result + "'" +  "title='" + picFile.name + "'/>";

//                     output.insertBefore(div,null);

//                     const webcams = document.getElementById("im"+ lm.toString());

//                     const webcamImages = tf.browser.fromPixels(webcams);

//                     const reversedImages = webcamImages.reverse(1);

//                     const size = Math.min(reversedImages.shape[0], reversedImages.shape[1]);
// 				    const centerHeight = reversedImages.shape[0] / 2;
// 				    const beginHeight = centerHeight - (size / 2);
// 				    const centerWidth = reversedImages.shape[1] / 2;

// 				    const beginWidth = centerWidth - (size / 2);
// 				    const croppedImage= reversedImages.slice([beginHeight, beginWidth, 0], [size, size, 3]);
// 				    const batchedImage = croppedImage.expandDims(0);

// 				    const final_img = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
// 				    predict(final_img);
				   
//                 });
//                 //Read the image
//                 picReader.readAsDataURL(file);
                
//             }
//         });
//  	}

//     else
//     {
//         console.log('Your browser does not support File API');
//     }
// }

function doTraining(){
	train();
}

async function init(){
	mobilenet = await loadMobilenet();
    document.getElementById("model_loaded").innerHTML="Model Has been , Move further and upload images";
}
    // console.log(mobilenet);




init();
