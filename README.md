# NeuralNetwork

This code implements a neural network for a class exercise (but the network package is for general use).

The docs for the package are inside the package/docs folder

Compilation, docs generation and execution by the following command lines (all executed in the src folder):

*Command line to generate the docs:
	javadoc -d cristovaotrevisan/neuralnetwork/docs cristovaotrevisan.neuralnetwork
*Command line to compile the code:
	* Compile the NeuralNode.java:
	javac cristovaotrevisan/neuralnetwork/NeuralNode.java
	* Compile the ActivatonType.java:
	javac cristovaotrevisan/neuralnetwork/ActivationType.java
	* Compile the Main file (and the NeuralNetwork.java by association)
	javac Main.java
*Command line to execute program:
	java Main


The Main file executes the exercise, which is (it's in portuguese because I'm lazy and that's' it):



No processamento de bebidas, a aplicação de um determinado conservante e feita em função da combinação de quatro variáveis de tipo real, definidas por x1 (teor da água), x2 (grau de acidez), x3 (temperatura) e x4 (tensão superficial). Sabe-se que existem apenas três tipos de conservantes que podem ser aplicados, os quais são definidos por A, B e C. Em seguida, realizam-se ensaios em laboratório a fim de especificar qual tipo deve ser aplicado em uma bebida específica.

A partir de 148 ensaios executados em laboratório, a equipe de engenheiros e cientistas resolveu aplicar uma rede Perceptron Multicamadas como classificadora de padrões, visando identificar qual tipo de conservante seria introduzido em determinado lote de bebidas. Por questões operacionais da própria linha de produção, utilizar-se-á  uma Perceptron com três saídas e 15 nós na única camada escondida.
   A padronização para a saída, a qual representa o conservante a ser aplicado, ficou definida de acordo com a tabela a seguir:

Tipo de Conservante	y1	y2	y3
Tipo A	1	0	0
Tipo B	0	1	0
Tipo C	0	0	1
    Utilizando os dados de treinamento disponível no arquivo de treinamento, execute então o treinamento de uma rede PMC da figura apresentada que possa classificar, em função apenas dos valores medidos de x1, x2, x3 e x4, qual o tipo de conservante que pode ser aplicado em determinada bebida. Para tanto faça as seguintes atividades:

   Utilize a taxa de aprendizado no valor de 0.1

   Para testar a sua rede, utilize os dados de teste. 

