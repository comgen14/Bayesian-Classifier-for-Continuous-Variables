//Daniel Mitchell
//CS 4375
//Final Project - Final Iteration
//12/15/18

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class Main {
	public static void main(String[] args) throws IOException{
		ArrayList<String[]> files = GetFiles("C:\\Users\\Daniel\\Documents\\Programming Projects\\Java\\Workspace\\CS4375FINAL\\src\\files.txt");
		ArrayList<String[]> data = GetData(files.get(0)[1]);
		ArrayList<String[]> test = GetData(files.get(1)[1]);
		ArrayList<String[]> classValues = GetData(files.get(2)[1]);
		ArrayList<String[]> attributes = GetData(files.get(3)[1]);
		double limit = Double.parseDouble(files.get(4)[1]);
		
		//breaks the data into the two different classes
		ArrayList<String[]> data0 = new ArrayList<String[]>();
		ArrayList<String[]> data1 = new ArrayList<String[]>();
		
		for(int i = 0; i < data.size(); i++){
			if(data.get(i)[data.get(0).length-1].compareTo("0") == 0){
				data0.add(data.get(i));
			}else{
				data1.add(data.get(i));
			}
		}
		
		//rotates the data so that each row is an attribute value for each data point
		Double[] cvr = RotateData(classValues).get(0);
		ArrayList<Double[]> testr = RotateData(test);
		ArrayList<Double[]> datar = RotateData(data);
		ArrayList<Double[]> data0r = RotateData(data0);
		ArrayList<Double[]> data1r = RotateData(data1);
		Double[] tcvr = datar.get(datar.size()-1);
		datar.remove(datar.size()-1);
		
		ArrayList<Double[]> ccp0 = GenerateMeanVarianceTable(data0r, attributes);
		ArrayList<Double[]> ccp1 = GenerateMeanVarianceTable(data1r, attributes);

		//usingPrintWriter(datar);
		
		boolean[] ignore = PrintCorrelationArray(datar , limit);
		
		//for now, we're not going to test around values where the Gaussian Probability Density function gives us values greater than 1
			double trainingSet = NaiveBayesTesting(datar, datar, ccp0, ccp1, attributes, tcvr, data0r, data1r, ignore);
			System.out.format("Accuracy on training set (%d instances): %.5f%%%n", data.size(), trainingSet * 100);
			double testSet = NaiveBayesTesting(testr, datar, ccp0, ccp1, attributes, cvr, data0r, data1r, ignore);
			System.out.format("Accuracy on test set (%d instances): %.5f%%%n", test.size(), testSet * 100);
		
		
	}

	public static boolean[] PrintCorrelationArray(ArrayList<Double[]> data, double limit) throws IOException{
		boolean[] ignore = new boolean[data.size()];
		for(int i = 0; i < ignore.length; i++){
			ignore[i] = false;
		}
		int[] pairsize = new int[data.size()];
		double[][] matrix = new double[data.size()][data.size()];
		FileWriter fileWriter = new FileWriter("highrsquared.txt");
	    PrintWriter printWriter = new PrintWriter(fileWriter);
	    
		for(int x = 0; x < data.size(); x++){
			for(int y = 0; y < data.size(); y++){
				if(y != x){
					matrix[x][y] = Math.pow(correlation(data.get(x), data.get(y)), 2);
				}
			}
		}
		
		for(int x = 0; x < data.size(); x++){
			pairsize[x] = 0;
			for(int y = 1; y < data.size(); y++){
				if(matrix[x][y] >= limit && ignore[x] == false){
					printWriter.format("(%d, %d)", x, y);
					printWriter.println();
					ignore[y] = true;
				}
			}
		}
		
		 printWriter.close();
		 
		 return ignore;
	}
	
	public static double correlation(Double[] datax, Double[] datay){
		double answer = 0.0;
		double meanx = 0;
		double meany = 0;
		int nx = 0;
		int ny = 0;
		double suma = 0;
		double sumb = 0;
		double sumab = 0;
		
		for(int i = 0; i < datax.length; i++){
			if(datax[i] != null){
				meanx += datax[i];
				nx++;
			}
			
			if(datay[i] != null){
				meany += datay[i];
				ny++;
			}
		}
		
		meanx = meanx/nx;
		meany = meany/ny;
		
		for(int i = 0; i < datax.length; i++){
			if(datax[i] != null && datay[i] != null){
				suma += Math.pow(datax[i] - meanx, 2);
				sumb += Math.pow(datay[i] - meany, 2);
				sumab += (datax[i] - meanx) * (datay[i] - meany);
			}
		}
		
		answer = sumab / Math.sqrt(suma *sumb);
		
		return answer;
	}
	
	public static ArrayList<String[]> GetData(String file) throws IOException{
		ArrayList<String[]> answer = new ArrayList<String[]>();
		String [] placeholder;
		String temp;
		File f = new File(file);
		BufferedReader br = new BufferedReader(new FileReader(f));
        while ((temp = br.readLine()) != null) {
			placeholder = temp.split(" ");
			answer.add(placeholder);
		}
		
		br.close();
		
		return answer;
		
	}
	
	public static ArrayList<String[]> GetFiles(String file) throws IOException{
		ArrayList<String[]> answer = new ArrayList<String[]>();
		String [] placeholder;
		String temp;
		File f = new File(file);
		BufferedReader br = new BufferedReader(new FileReader(f));
        while ((temp = br.readLine()) != null) {
			placeholder = temp.split("\t");
			answer.add(placeholder);
		}
		
		br.close();
		
		return answer;
		
	}
	//converts the columns of attributes into rows of doubles. '?' are initialized to NULL 
	public static ArrayList<Double[]> RotateData(ArrayList<String[]> data){
		ArrayList<Double[]> answer = new ArrayList<Double[]>() ;
		Double[] temp =  new Double[data.size()];
		for(int j = 0; j < data.get(0).length; j++){
			temp =  new Double[data.size()];
			for(int i = 0; i < data.size(); i++){
				if(!data.get(i)[j].equals("?"))
					temp[i] = Double.parseDouble(data.get(i)[j]);
			}
			answer.add(temp);
		}
		
		
		return answer;
	}
	//This function generates a Continuous Class Conditional Prior as found in a Guassian Naive Bayesian Classifier https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
	//takes a mean standard deviation tuple and returns the probability that the value v is true given the corresponding class value.
	public static double CCCP(double v, double mean, double variance){
		double answer = 0;
		
		answer = (1 / Math.sqrt(2 * Math.PI * variance)) * Math.exp(-1 * Math.pow(v - mean, 2) / (2 * variance));
		
		return answer;
	}
	
	//this function calculates the mean and variance of a section of data and returns it as a mean/variance tuple.
	public static Double[] MeanVariance(Double[] data){
		double mean = 0;
		int mlength = 0;
		double variance = 0;
		Double[] answer;
		//compute mean
		for(int i = 0; i< data.length; i++){
			if(data[i] != null)
				mean += data[i];
				mlength++;
		}
		mean = mean / mlength;
		//compute variance
		for(int i = 0; i< data.length; i++){
			if(data[i] != null){
				variance += Math.pow((mean - data[i]), 2);
			}
		}
		variance = variance / mlength;
		
		answer = new Double[] {mean, variance};
		
		return answer;
		
	}
	//This generates our Conditional Class Prior. Takes an attributes value, v, and the corresponding sorted data set to generate the CCP. 
	//this function is meant to be used on demand so that we can easily calculate discrete values that are larger than size {0, 1}, i.e., {0, 1, 2, 3, 4, 5}.
	public static Double GetCCP(double v, Double[] data){ 
		Double conProb = 0.0;
		int count = 0;
		int sumOfClass = 0;
			for(int i = 1; i < data.length; i++){
				if(data[i] != null){
					sumOfClass++;
					if(data[i] == 1){
						count++;
					}
				}
			}
			conProb = (double) (count) /(double) (sumOfClass);

		return conProb;
	}
	//generate a table of mean variance tuples for quick use later
	public static ArrayList<Double[]> GenerateMeanVarianceTable(ArrayList<Double[]> data, ArrayList<String[]> attr){
		ArrayList<Double[]> answer = new ArrayList<Double[]>();
		
		for(int i = 0; i < data.size()-1; i++){
			if(attr.get(i)[1].equals("cont")){
				answer.add(MeanVariance(data.get(i)));
			}else{
				answer.add(new Double[] {});
			}
		}
		
		return answer;
	}
	
	public static void usingPrintWriter(ArrayList<Double[]> data) throws IOException
	{
	    FileWriter fileWriter = new FileWriter("output.txt");
	    PrintWriter printWriter = new PrintWriter(fileWriter);
	    for(int i = 0; i < data.size(); i++){
	    	for(int j = 0; j < data.get(i).length; j++)
	    		printWriter.format("%.2f ", data.get(i)[j]);
	    	printWriter.println();
	    }
	    
	    printWriter.close();
	}
	
	public static double NaiveBayesTesting(ArrayList<Double[]> test, ArrayList<Double[]> data, ArrayList<Double[]> ccp0, ArrayList<Double[]> ccp1, ArrayList<String[]> attr, Double[] cvr, ArrayList<Double[]> data0, ArrayList<Double[]> data1, boolean[] ignore) throws IOException {
		int correctness = 0;
		double p0 = Math.log((double) (ccp0.size())/(double) (test.size())); //take the log of the probabilities to prevent underflow
		double p1 = Math.log((double) (ccp1.size())/(double) (test.size()));
		double temp0 = 0;
		double temp1 = 0;
		double[] arrayTemp = {p0, p1};
		
		FileWriter fileWriter = new FileWriter("prediction.txt");
	    PrintWriter printWriter = new PrintWriter(fileWriter);
	    FileWriter fileWriter2 = new FileWriter("redundant.txt");
	    PrintWriter printWriter2 = new PrintWriter(fileWriter2);
		
		//iterates through each column. Columns are data points
		for(int x = 0; x < test.get(0).length; x++){
			//takes the log sums of the corresponding probabilities
			//the for loop iterates through each row. A row represents an attribute of a data point
			//x iterates for data points. y iterates for attributes
			for(int y = 0; y < test.size(); y++){
				if(test.get(y)[x] != null && ignore[y] == false){//check to make sure there is a variable to compare against, ignores attributes that have high correlation with attributes we want to test on.
					if(attr.get(y)[1].equals("cont")){ //if the value is continuous, use the Gaussian
						temp0 = Math.log(CCCP(test.get(y)[x], ccp0.get(y)[0], ccp0.get(y)[1]));
						temp1 = Math.log(CCCP(test.get(y)[x], ccp1.get(y)[0], ccp1.get(y)[1]));
					}else{ //if the value is binary, use the probability from the GetCCP function
						temp0 = Math.log(GetCCP(test.get(y)[x], data0.get(y)));
						temp1 = Math.log(GetCCP(test.get(y)[x], data1.get(y)));
					}
					if(!Double.isInfinite(temp0) && !Double.isNaN(temp0) && !Double.isInfinite(temp1) && !Double.isNaN(temp1)){
						arrayTemp[0] += temp0;
						arrayTemp[1] += temp1;
					}
				}

			}
			printWriter2.format("%.4f, %.4f", arrayTemp[0], arrayTemp[1]);
			printWriter2.println();
			//compares to decide if the guess is the same as the actual class value
			if(arrayTemp[0] > arrayTemp[1]){
				printWriter.println("0");
				if(cvr[x] == 0.0){
					correctness++;
				}	
			}else if(arrayTemp[0] <= arrayTemp[1]){
				printWriter.println("1");
				if(cvr[x] == 1.0){
					correctness++;
				}
			}
			//reset the array
			
			arrayTemp[0] = p0;
			arrayTemp[1] = p1;
		}
		
		printWriter.close();
		printWriter2.close();
		
		return (double)correctness/(double)(test.get(0).length);
	}

}
