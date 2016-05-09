package cristovaotrevisan.neuralnetwork;

class NeuralNode {
	
	// node offset and weigths
	private float b, w[];
	// node registers
	private float lastInput[], lastOutput;
	// activation type
	private ActivationType activationType;

	// this makes calculations much faster
	private float exp(float x) {
		x = 1f + x / 256f;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x; x *= x; x *= x;
		return x;
	}

	/**
		@param w = initial weigths
		@param b = initial offset
	*/
	NeuralNode(float w[], float b, ActivationType activationType){
		this.w = w;
		this.b = b;
		this.activationType = activationType;
		this.lastOutput = 0;
		this.lastInput = new float[0];
	};

	// gets and setters
	public float[] getInput(){
		return lastInput;
	}

	public float getOutput(){
		return lastOutput;
	}

	public float[] getWeigths(){
		return w;
	}

	public float getWeigth(int i){
		return w[i];
	}

	public float getOffset(){
		return b;
	}

	public void setWeigths(float[] w){
		this.w = w;
	}

	public void setOffset(float b){
		this.b = b;
	}


	/**
		Special setter to change only the gain of a certain input
		@param w = new weigth
		@param i = index of weigth to be changed
	*/
	public void setWeigth(float w, int i) throws Exception{
		if(i < 0 || i>=this.w.length)
			throw new Exception("Index out of bound");
		this.w[i] = w;
	}


	public float calculate(float input[]) throws Exception{
		if (input.length != w.length)
			throw new Exception("Wrong number of inputs for NeuralNode");
		lastInput = input;
		float out = 0;
		for (int i=0; i<w.length; i++)
			out += w[i]*input[i];
		switch(activationType){
			case SIGMOID:
				lastOutput = 1f/(1f+exp(-out-b));
				return lastOutput;
			case STEP:
				lastOutput = (out+b) > 0.5f ? 1 : 0;
				return lastOutput;
			default:
				lastOutput = 0;
				return 0;
		}
	}

}