/*
 * Macro to fix position index shift
 * maybe add autoscaling (reference roi set by user?)
 */

#@ File (label = "Directory of preprocessed files", style = "directory") dir
#@ File (label = "Directory for output files", style = "directory") outdir
#@ String (label = "Output file format", value = ".ome.tf2") format

setBatchMode(true);

list = getFileList(dir);
list = Array.sort(list);
for (i = 1; i < list.length; i++) {
	if(endsWith(list[i], format))
		print("");
		print(i +" out of " + list.length);
		print("    "+ list[i]);
		fixShift(list, i, dir);
}

function fixShift(list, i, dir) {
	stack1 = list[i];
	stack2 = list[i-1];
	//stack3 = list[i-2];
	// open first
	bioFormatsString = "open=" + dir + File.separator + stack1 + " color_mode=Default view=Hyperstack stack_order=XYCZT";
	run("Bio-Formats Importer", bioFormatsString);
	id1 = getImageID();
	run("Make Substack...", "channels=1-2 frames=1-270");
	rename("one");
	selectImage(id1);
	close();
	// open second
	bioFormatsString = "open=" + dir + File.separator + stack2 + " color_mode=Default view=Hyperstack stack_order=XYCZT";
	run("Bio-Formats Importer", bioFormatsString);
	id2 = getImageID();
	run("Make Substack...", "channels=1-2 frames=271-328");
	rename("two");
	selectImage(id2);
	close();
	// open third
	//bioFormatsString = "open=" + dir + File.separator + stack3 + " color_mode=Default view=Hyperstack stack_order=XYCZT";
	//run("Bio-Formats Importer", bioFormatsString);
	//id3 = getImageID();
	//run("Make Substack...", "channels=1-2 frames=97-187");
	//rename("three");
	//selectImage(id3);
	//close();
	// merge sub_stacks
	//run("Concatenate...", "  title=fixed open image1=one image2=two image3=three");
	run("Concatenate...", "  title=fixed open image1=one image2=two");
	outPath = outdir + File.separator + "corr_" + stack1 + format;
	saveFile(outPath);
	run("Close All");
	run("Collect Garbage");
}

function saveFile(outPath) {
   run("Bio-Formats Exporter", "save=[" + outPath + "] compression=Uncompressed");
}