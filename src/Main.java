import cristovaotrevisan.neuralnetwork.*;

public class Main{
	public static void main(String args[]){
		try{
			testBinyraOrFromFile();
		}
		catch (Exception e){
			e.printStackTrace();
		}
	}

	public static void workOnHomework() throws Exception{
		// creates neural network with:
		//	4 inputs
		//	Sigmoid activation function
		// 15 nodes on the first hidden layer
		// 3 nodes on the output layer
		NeuralNetwork n = new NeuralNetwork(ActivationType.SIGMOID, 4, 15, 3);

		float tolerance = 0.3f;
		// train using input from file with 0.2 learning rate and 10000 epochs
		n.trainInputFromFile("../training_cases.in", 0.1f, 10000, tolerance);

		// test cases, print the tests and error count (with 0.2 absolute tolerance)
		n.printTestFromFile("../test_cases.in");
		System.out.format("Error Count with %.4f tolerance:\n", tolerance);
		System.out.println(n.testInputFromFile("../test_cases.in", tolerance));
	}

	public static void testBinyraOrFromFile() throws Exception{
		NeuralNetwork n = new NeuralNetwork(ActivationType.SIGMOID, 2, 2, 1);
		float tolerance = 0.01f;
		n.trainInputFromFile("../binary_or.in", 3f, 10000, tolerance);

		n.printTestFromFile("../binary_or.in");
		System.out.format("Error Count with %.4f tolerance:\n", tolerance);
		System.out.println(n.testInputFromFile("../binary_or.in", tolerance));
	}


	public static void testBinaryOp() throws Exception{
		NeuralNetwork n = new NeuralNetwork(ActivationType.SIGMOID, 2, 2, 1);
		float input1[]  = {0f, 0f};
		float expected1[] = {0f};
		float input2[]  = {0f, 1f};
		float expected2[] = {0f};
		float input3[]  = {1f, 0f};
		float expected3[] = {1f};
		float input4[]  = {1f, 1f};
		float expected4[] = {1f};
		n.print();
		System.out.println(n.calculate(input1)[0]);
		System.out.println(n.calculate(input2)[0]);
		System.out.println(n.calculate(input3)[0]);
		System.out.println(n.calculate(input4)[0]);
		System.out.println("----------------------------");
		for(int i=0; i<10000; i++){
			n.trainInput(input1, expected1, 0.5f);
			n.trainInput(input2, expected2, 0.5f);
			n.trainInput(input3, expected3, 0.5f);
			n.trainInput(input4, expected4, 0.5f);
		}
		System.out.println("----------------------------");
		n.print();
		System.out.println(n.calculate(input1)[0]);
		System.out.println(n.calculate(input2)[0]);
		System.out.println(n.calculate(input3)[0]);
		System.out.println(n.calculate(input4)[0]);
	}
}