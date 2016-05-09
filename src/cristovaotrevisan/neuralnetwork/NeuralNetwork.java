package cristovaotrevisan.neuralnetwork;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.io.File;

public class NeuralNetwork {
	// holds the neural nodes network (with values)
	private ArrayList<ArrayList<NeuralNode>> network;
	private int numOfInputs;
	private int numOfOutputs;
	private ActivationType activationType;
	/**
		@param layers Number of layers
		@param layerSize Number of nodes in each layer
	*/
	public NeuralNetwork(ActivationType activationType, int numOfInputs, int...layerSize) throws Exception{ 
		network = new ArrayList<ArrayList<NeuralNode>>();
		this.numOfInputs = numOfInputs;
		this.numOfOutputs = layerSize[layerSize.length-1];
		this.activationType = activationType;
		// initialize nodes with random values
		Random rand = new Random();
		for (int i=0; i<layerSize.length; i++){
			// buff for current layer
			ArrayList<NeuralNode> layerBuff = new ArrayList<NeuralNode>(layerSize[i]);
			// weigths generation
			float ws[][] = new float[layerSize[i]][(i==0) ? numOfInputs : layerSize[i-1]];
			float sum = 0;
			for (int j=0; j<layerSize[i]; j++)
				for(int k=0; k<((i==0) ? numOfInputs : layerSize[i-1]); k++) {
					// -1 to 1
					ws[j][k] = rand.nextFloat()*2-1;
					sum += ws[j][k];
				}
			// offsets generation (must sum 0 with ws)
			for (int j=0; j<layerSize[i]; j++){
				//float bs = (j==(layerSize[i]-1)) ? -sum : -sum*rand.nextFloat();
				float bs = rand.nextFloat()*2-1;
				sum += bs;
				// crete node
				layerBuff.add(new NeuralNode(ws[j], bs,activationType));
			}
			// adds layer to network
			network.add(layerBuff);
		}

	};

	/**
		Prints out all nodes in the network, in the form:
		W(l)ij, where:
		- l is the layer
		- i is the number of the node
		- j is the number of the connection (number of the input for the first layer)
	*/
	public void print(){
		for (int i=0; i<network.size(); i++){
			ArrayList<NeuralNode> layerBuff = network.get(i);
			for(int j=0; j<layerBuff.size(); j++)
				for(int k=0; k<layerBuff.get(j).getWeigths().length; k++)
					System.out.format("w(%d)%d%d: (%.3f, %.3f)\n", i+1, k+1, j+1, layerBuff.get(j).getWeigths()[k], layerBuff.get(j).getOffset());
		}
	}

	/**
		Calculate network outputs for the inputs parameter (also updates values in network, but not
		weigths or bias)
	*/
	public float[] calculate(float input[]) throws Exception{
		if (input.length != numOfInputs)
			throw new Exception("Wrong number of inputs for NeuralNode");
		
		// for each layer
		float results[][] = new float[network.size()][];
		for(int i=0; i<network.size(); i++){
			float layerInput[] = (i==0 ? input : results[i-1]);
			// for each node
			results[i] = new float[network.get(i).size()];
			for (int j=0; j<network.get(i).size(); j++){
				results[i][j] = network.get(i).get(j).calculate(layerInput);
			}
		}
		return results[network.size()-1];

	}

	// calculate the derivate for the actvation function (only sigmoid working)
	private float derivativeCalculation(float x){
		switch(activationType){
			case SIGMOID:
				return x * (1-x);
			case STEP:
				return (x>-0.05f && x<0.05f) ? 10 : 0;
			default:
				return 0;
		}
	}

	/**
		Training function to a single input
		@param input Inputs
		@param expectedValue Object value to be achieved
		@param rate Learning rate
	*/
	public void trainInput(float input[], float[] expectedValue, float rate) throws Exception{
		// calculate output (feedfoward)
		float[] outputValue = this.calculate(input);

		// create variable to hold deltas
		float delta[][] = new float[network.size()][];
		delta[network.size()-1] = new float[numOfOutputs];
		// for each output node
		for(int i=0; i<numOfOutputs; i++){
			// calculate error factor
			float errorFactor = expectedValue[i] - outputValue[i];
			// set delta
			delta[network.size()-1][i] =  derivativeCalculation(outputValue[i]) * errorFactor;
		}
		// for each layer but the output layer
		for(int i=network.size()-2;i>=0; i--){
			// initialize delta for layer
			delta[i] = new float[network.get(i).size()];
			// for each node in the layer
			for(int j=0; j<delta[i].length; j++){
				float errorFactor = 0;
				// for each output connection
				for(int k=0; k<delta[i+1].length; k++){
					errorFactor += delta[i+1][k] * network.get(i+1).get(k).getWeigth(j);
				}
				delta[i][j] = derivativeCalculation(network.get(i).get(j).getOutput()) * errorFactor;
			}
		}

		// for each layer
		for(int i=0; i<network.size(); i++){
			// for each node in layer
			for(int j=0; j<network.get(i).size(); j++){
				// update offset (bias value)
				float oldOffset = network.get(i).get(j).getOffset();
				network.get(i).get(j).setOffset(oldOffset + rate * delta[i][j]);

				// update weigths
				float oldW[] = network.get(i).get(j).getWeigths();
				float inp[] = network.get(i).get(j).getInput();
				// for each input of the nodes
				for(int k=0; k<oldW.length; k++){
					network.get(i).get(j).setWeigth(oldW[k] + rate * inp[k] * delta[i][j] ,k);
				}
			}
		}
	}

	/**
		Train the neural network for the cases in file (each line should have m inputs and n outputs,
		m and n being set in the construtor, also the first line is ignored- it's your header)
		@param fileName Path to file
		@param rate Learning rate
		@param times Number of epochs
	*/
	public void trainInputFromFile(String fileName, float rate, int times) throws Exception{
		Scanner scan;
		File file = new File(fileName);
		scan = new Scanner(file);

		// get number of training sets
		int count = -1;
		while (scan.hasNextLine()) {
		    count++;
		    scan.nextLine();
		}

		// read training sets
		scan = new Scanner(file);
		scan.nextLine();
		float input[][] = new float[count][numOfInputs];
		float output[][] = new float[count][numOfOutputs];
		int a=0;

    	for (int i=0; i<count; i++){
    		// read inputs
    		for (int j=0; j<numOfInputs; j++)
    			input[i][j] = scan.nextFloat();
    		// read outputs
    		for (int j=0; j<numOfOutputs; j++)
    			output[i][j] = scan.nextFloat();
    	}

    	for (int k=0; k<times; k++){
	    	for (int i=0; i<count; i++)
	    		this.trainInput(input[i], output[i], rate);	
	    	System.out.format("Training: %d%%\r", k*100/times);
	    }
	    System.out.println("Training: 100%");
	};

	/**
		This function returns the number of errors found for the tests in the file (same rules as in training), with tolerance (+/-)
		@param fileName Path of the file to be read
		@param tolerance absolute tolerance to count error
	*/
	public int testInputFromFile(String fileName, float tolerance) throws Exception{
		Scanner scan;
		File file = new File(fileName);
		scan = new Scanner(file);

		// get number of tests
		int count = -1;
		while (scan.hasNextLine()) {
		    count++;
		    scan.nextLine();
		}

		// read tests sets
		scan = new Scanner(file);
		scan.nextLine();
		float input[][] = new float[count][numOfInputs];
		float output[][] = new float[count][numOfOutputs];
		int a=0;

    	for (int i=0; i<count; i++){
    		// read inputs
    		for (int j=0; j<numOfInputs; j++)
    			input[i][j] = scan.nextFloat();
    		// read outputs
    		for (int j=0; j<numOfOutputs; j++)
    			output[i][j] = scan.nextFloat();
    	}

    	int errorCount = 0;
    	for (int i=0; i<count; i++){
    		float calcOut[] = this.calculate(input[i]);
    		for (int j=0; j<numOfOutputs; j++){
    			if ( calcOut[j]< output[i][j]-tolerance || calcOut[j]> output[i][j]+tolerance)
    				errorCount++;

    			//System.out.format("o%d: %.4f\t", j, calcOut[j]);
	    	}
	    	//System.out.println();
		}
	    return errorCount;
	};

	/**
		This function prints out the results for each test in the file (same rules as in training).
		Each line is a test case, with the outputs in order
		@param fileName Path of the file to be read
		@param tolerance absolute tolerance to count error
	*/
	public void printTestFromFile(String fileName) throws Exception{
		Scanner scan;
		File file = new File(fileName);
		scan = new Scanner(file);

		// get number of tests
		int count = -1;
		while (scan.hasNextLine()) {
		    count++;
		    scan.nextLine();
		}

		// read tests sets
		scan = new Scanner(file);
		scan.nextLine();
		float input[][] = new float[count][numOfInputs];
		float output[][] = new float[count][numOfOutputs];
		int a=0;

    	for (int i=0; i<count; i++){
    		// read inputs
    		for (int j=0; j<numOfInputs; j++)
    			input[i][j] = scan.nextFloat();
    		// read outputs
    		for (int j=0; j<numOfOutputs; j++)
    			output[i][j] = scan.nextFloat();
    	}

    	int errorCount = 0;
    	for (int i=0; i<count; i++){
    		float calcOut[] = this.calculate(input[i]);
    		for (int j=0; j<numOfOutputs; j++){
    			System.out.format("o%d: %.4f\t", j, calcOut[j]);
	    	}
	    	System.out.println();
		}
	};
}