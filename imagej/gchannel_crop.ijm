#@ File (label = "Select tif file", style = "file") input
#@ File (label = "Directory for output files", style = "directory") outdir
#@ String (label = "Output file format", value = ".ome.tf2") format

bioFormatsString = "open=" + input + " color_mode=Default view=Hyperstack stack_order=XYCZT";
run("Bio-Formats Importer", bioFormatsString);

name =  getTitle();
name_arr = split(name, ".");
name_fixed = name_arr[0];

Stack.setChannel(1);
run("Grays");
Stack.setChannel(2);
run("Green");
Stack.setChannel(1);
run("Enhance Contrast", "saturated=0.35");

run("Select None");
setTool("multipoint");
waitForUser("Please select bottom of gchannels of interest. Click OK when done")
run("Clear Results");
run("Measure");

//ask user to specify channel numbers
setOption("ExpandableArrays", true);
list = newArray;

for (i=0; i<nResults; i++) {
	qString = "Channel number of point Nr." + i + "?";
	list[i] = getNumber(qString, 0);
}

//crop loop
//setBatchMode(true);
for (i=0; i<nResults; i++) {
	wait(100);
	selectWindow(name);
	wait(100);
	run("Duplicate...", "title=crop duplicate");
	wait(100);
	px = getResult("X",i);
	py = getResult("Y",i);
	toUnscaled(px); // make sure coordinates are in pixels
	toUnscaled(py);
	xLength = 200;
	yLength = 500;
	// mother cell down gchannels
	//makeRectangle(px-(xLength/2), (py-yLength+50), xLength, yLength); //the coordinates index from the top left, like a 2D array
	// mother cell up gchannels
	makeRectangle((px-(xLength/2)), (py-100), xLength, yLength); //the coordinates index from the top left, like a 2D array
	wait(100);
	run("Crop");
	wait(100);
	setMinAndMax(0, 65535);
	wait(100);
	outPath = outdir + File.separator + name_fixed + "_gch" + list[i] + format;
	saveFile(outPath);
}

wait(1000);
selectWindow("Results");
run("Close"); 
run("Close All");
run("Collect Garbage");

function saveFile(outPath) {
   run("Bio-Formats Exporter", "save=[" + outPath + "] compression=Uncompressed");
}