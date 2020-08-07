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
Ext.getSizeZ(sizeZ);
Ext.getSizeT(sizeT);
Ext.close();
// split by Z, C and T, only if necessary
halfT = floor(sizeT/2);

// iterate over positions (Z)
for (currZ=1; currZ<sizeZ+1; currZ++) {
	print("Processing " + currZ + " out of " + sizeZ);
	for (currC=1; currC<3; currC++) {
		bioFormatsString = "open=" + input + " color_mode=Default specify_range view=Hyperstack stack_order=XYCZT c_begin_1=" + currC + " c_end_1=" + currC + " c_step_1=1 z_begin_1=" + currZ + " z_end_1=" + currZ + " z_step_1=1";
		print(bioFormatsString);
		if (currC==1) {
			cName = "FI";
		} else {
			cName = "BF";
		}
		outPath = output + File.separator + prefix + IJ.pad(currZ,2) + "_" + cName + format;
		processSubStack(bioFormatsString, outPath);
		run("Close All");
		run("Collect Garbage");
	}
}
print("Finished."); 

function processSubStack(bioFormatsString, outPath) {
	run("Bio-Formats Importer", bioFormatsString);
	// actual processing
	makeRectangle(trx, try, blx, bly);
	run("Crop"); 
	setMinAndMax(0, 65535);
	//run("8-bit");
	//run("Bin...", "x=2 y=2 z=1 bin=Average");
	// saving
	saveFile(outPath);
}

function saveFile(outPath) {
   run("Bio-Formats Exporter", "save=[" + outPath + "] export compression=Uncompressed");
}