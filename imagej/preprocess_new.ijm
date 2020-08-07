/*
 * Macro to preprocess nd2 stacks
 * Output: One .tif stack per position
 */

#@ File (label = "Select nd2 file", style = "file") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "Output file prefix", value = "pos") prefix
#@ String (label = "Output file format", value = ".tif") format
#@ int (label = "Crop: Top left x =", value = "0") trx
#@ int (label = "Crop: Top left y =", value = "0") try
#@ int (label = "Crop: Bottom right x =", value = "0") blx
#@ int (label = "Crop: Bottom right y =", value = "0") bly

//doCommand("Monitor Memory..."); 
//run("Conversions...", " ");
run("Conversions...", "scale");
setBatchMode(true);
run("Bio-Formats Macro Extensions");
Ext.setId(input);
Ext.getSeriesCount(nSeries);
Ext.close();

// iterate over positions (series)
for (i=0; i<nSeries; i++) {
	series = i + 1;
	outPath = output + File.separator + prefix + IJ.pad(series,2) + format;
	bioFormatsString = "open=" + input + " color_mode=Default view=Hyperstack stack_order=XYCZT series_" + series;
	print("Processing " + series + " out of " + nSeries);
	processSeries(bioFormatsString, outPath);
	run("Close All");
	run("Collect Garbage");
}

print("Finished.")

function processSeries(bioFormatsString, outPath) {
	run("Bio-Formats Importer", bioFormatsString);
	// actual processing
	run("Rotate... ", "angle=7 grid=1 interpolation=Bilinear stack"); // to account for non vertical gchannels
	makeRectangle(trx, try, blx, bly);
	run("Crop"); 
	setMinAndMax(0, 65535);
	//run("8-bit");
	//run("Bin...", "x=2 y=2 z=1 bin=Average");
	// saving
	saveFile(outPath);
}

function saveFile(outPath) {
   run("Bio-Formats Exporter", "save=[" + outPath + "] compression=Uncompressed");
}

// run("Bio-Formats", "open=[input] autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");