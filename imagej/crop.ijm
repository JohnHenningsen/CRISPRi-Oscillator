/*
 * Macro template to process multiple images in a folder
 */

#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".tif") suffix
#@ int (label = "Crop: Top left x =", value = "0") trx
#@ int (label = "Crop: Top left y =", value = "0") try
#@ int (label = "Crop: Bottom right x =", value = "2560") blx
#@ int (label = "Crop: Bottom right y =", value = "2160") bly
#@ int (label = "Number of time points =", value = 382) sizeT

run("Conversions...", "scale");
setBatchMode(true);

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		curr = i+1;
		print(curr + " out of " + list.length);
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	//print(file);
	// open file
	bioFormatsString = "open=" + input + File.separator + file + " color_mode=Default view=Hyperstack";
	run("Bio-Formats Importer", bioFormatsString);
	// process
	if (endsWith(file, "C0.tif")) {
		//print("FI, remove every 2nd frame");
		// FI only every 2nd timestep
		run("Slice Remover", "first=2 last=" + sizeT + " increment=2");
	}
	makeRectangle(trx, try, blx, bly);
	run("Crop"); 
	setMinAndMax(0, 65535);
	//run("8-bit");
	//run("Bin...", "x=2 y=2 z=1 bin=Average");
	// save file
	outPath = output + File.separator + "crop" + file;
	run("Bio-Formats Exporter", "save=[" + outPath + "] compression=Uncompressed");
	run("Close All");
	run("Collect Garbage");
}
