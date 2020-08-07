/*
 * Macro to split channels of tif
 */

#@ File (label = "Directory of preprocessed files", style = "directory") dir
#@ File (label = "Directory for output files", style = "directory") outdir
#@ String (label = "File format", value = ".tif") format
#@ Integer (label = "Number of channels", value = 2) countC

setBatchMode(true);

// loop over tifs in folder
list = getFileList(dir);
list = Array.sort(list);
// check sizeT for one example file, assume all are the same
run("Bio-Formats Macro Extensions");
examplePath = dir + File.separator + list[1];
Ext.setId(examplePath);
Ext.getSizeT(sizeT);
Ext.close();

for (i = 0; i < list.length; i++) {
	if(endsWith(list[i], format))
		print("");
		print(i +" out of " + list.length);
		print("    "+ list[i]);
		splitChannels(list, i);
		run("Close All");
		run("Collect Garbage");
}
print("");
print("");
print("DONE");


function splitChannels(list, i) {
	wait(100);
	bioFormatsString = "open=" + dir + File.separator + list[i] + " color_mode=Default view=Hyperstack stack_order=XYCZT";
	run("Bio-Formats Importer", bioFormatsString);
	id0 = getImageID();
	for (currC=1; currC<(countC+1); currC++) {
		wait(100);
		substackstring = "channels=" + currC + " frames=1-" + sizeT;
		run("Make Substack...", substackstring);
		wait(100);
		if (currC==1) {
			cName = "BF";
		} else {
			cName = "FI";
		}
		name_arr = split(list[i], ".");
		filename = name_arr[0] + "_" + cName + format;
		outPath = outdir + File.separator + filename;
		run("Bio-Formats Exporter", "save=[" + outPath + "] export compression=Uncompressed");
		wait(100);
		close();
		wait(100);
		selectImage(id0);
	}




}
