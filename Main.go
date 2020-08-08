package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"reflect"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("Implementing Linear Regression using Golang")
	fmt.Println("Reading CSV file..")
	file, err := os.Open("./src/github.com/HalfBloodPrince010/firstgoapp/data.csv")
	if err != nil {
		fmt.Println("Couldnt read the csv..there might be some problem!", err)
	}
	defer file.Close()
	data := csv.NewReader(file)
	data.FieldsPerRecord = 3
	dataRecords, err := data.ReadAll()
	if err != nil {
		fmt.Println("Problem reading the data!")
	}
	fmt.Println(reflect.TypeOf(dataRecords))
	fmt.Println("=====================SPLITTING DATA=========================")
	dataSplitter(dataRecords)
	fmt.Println("Done reading the data and creating the separate files..")
	fmt.Println("=====================TRAINING DATA=========================")
	trainFile, err := os.Open("./src/github.com/HalfBloodPrince010/firstgoapp/train.csv")
	if err != nil {
		fmt.Println("Couldnt read the train.csv..there might be some problem!", err)
	}
	defer trainFile.Close()
	trainData := csv.NewReader(trainFile)
	trainData.FieldsPerRecord = 3
	trainDataRecords, err := trainData.ReadAll()
	if err != nil {
		fmt.Println("Problem reading the data!")
	}
	weights := train(trainDataRecords)
	fmt.Println("=====================TEST DATA=========================")
	testFile, err := os.Open("./src/github.com/HalfBloodPrince010/firstgoapp/test.csv")
	if err != nil {
		fmt.Println("Couldnt read the test.csv..there might be some problem!", err)
	}
	defer testFile.Close()
	testData := csv.NewReader(testFile)
	testData.FieldsPerRecord = 3
	testDataRecords, err := testData.ReadAll()
	if err != nil {
		fmt.Println("Problem reading the data!")
	}
	predict(testDataRecords, weights)

}

func dataSplitter(data [][]string) {
	// Lets make it 0.8 - train and 0.2 - test
	var trainSplit float64 = 0.8
	trainLen := int(float64(len(data)) * trainSplit)
	fmt.Println(trainSplit)
	trainingSet := data[0:trainLen]
	testSet := data[trainLen+1:]
	fmt.Println(trainingSet)
	for idx, record := range trainingSet {
		fmt.Println("Record Train:", idx, "->", record)
	}
	dataSets := map[string][][]string{
		"./src/github.com/HalfBloodPrince010/firstgoapp/train.csv": trainingSet,
		"./src/github.com/HalfBloodPrince010/firstgoapp/test.csv":  testSet,
	}

	// Create a seperate file for test and train data set
	// we use os package to create the files.

	for filename, dataset := range dataSets {
		//Create creates or truncates the named file. If the file already exists, it is truncated. If the file does not exist, it is created with mode 0666 (before umask)
		file, err := os.Create(filename)
		if err != nil {
			fmt.Println("Problem creating the file..")
		}
		defer file.Close()
		dataWriter := csv.NewWriter(file)
		if err := dataWriter.WriteAll(dataset); err != nil {
			log.Fatal(err)
		}
		dataWriter.Flush()
	}

}

func train(trainData [][]string) mat.Dense {
	fmt.Println("Training the model..")
	oneDimArray := []float64{}
	oneDimArrayLabel := []float64{}
	for i := 0; i < len(trainData); {
		x, err := strconv.ParseFloat(trainData[i][0], 64)
		if err != nil {
			log.Fatal(err)
		}
		y, err := strconv.ParseFloat(trainData[i][1], 64)
		if err != nil {
			log.Fatal(err)
		}
		z, err := strconv.ParseFloat(trainData[i][2], 64)
		if err != nil {
			log.Fatal(err)
		}
		oneDimArray = append(oneDimArray, 1)
		oneDimArray = append(oneDimArray, x)
		oneDimArray = append(oneDimArray, y)
		oneDimArrayLabel = append(oneDimArrayLabel, z)
		i = i + 1
	}
	fmt.Println("Data Matrix")
	dataMatrix := mat.NewDense(len(trainData), 3, oneDimArray)
	// fmt.Println(dataMatrix)
	fmt.Println("Label Matrix")
	labelMatrix := mat.NewVecDense(len(trainData), oneDimArrayLabel)
	fmt.Println(labelMatrix)
	fmt.Println("==========================Calculating Weights===============================")
	var test mat.Dense
	test.Mul(dataMatrix.T(), dataMatrix)
	//Get the dimensions of the matrix A
	rows, cols := test.Dims()
	fmt.Println("TEST: rows: ", rows)
	fmt.Println("TEST: cols: ", cols)
	var ia mat.Dense
	// Take the inverse of a and place the result in ia.
	err := ia.Inverse(&test)
	if err != nil {
		log.Fatalf("a is not invertible: %v", err)
	}
	var peusdoInverse mat.Dense
	peusdoInverse.Mul(&ia, dataMatrix.T())
	var weights mat.Dense
	weights.Mul(&peusdoInverse, labelMatrix)
	fmt.Println("Weights")
	fmt.Println(weights)
	fmt.Println(reflect.TypeOf(weights))
	return weights
}

func predict(testData [][]string, weights mat.Dense) {
	fmt.Println("==========================PREDICTING===============================")
	fmt.Println("Testing the model..")
	oneDimArray := []float64{}
	oneDimArrayLabel := []float64{}
	for i := 0; i < len(testData); {
		x, err := strconv.ParseFloat(testData[i][0], 64)
		if err != nil {
			log.Fatal(err)
		}
		y, err := strconv.ParseFloat(testData[i][1], 64)
		if err != nil {
			log.Fatal(err)
		}
		z, err := strconv.ParseFloat(testData[i][2], 64)
		if err != nil {
			log.Fatal(err)
		}
		oneDimArray = append(oneDimArray, 1)
		oneDimArray = append(oneDimArray, x)
		oneDimArray = append(oneDimArray, y)
		oneDimArrayLabel = append(oneDimArrayLabel, z)
		i = i + 1
	}
	fmt.Println(" Test Data Matrix")
	testdataMatrix := mat.NewDense(len(testData), 3, oneDimArray)
	rows, cols := testdataMatrix.Dims()
	fmt.Println("TEST: rows: ", rows)
	fmt.Println("TEST: cols: ", cols)
	wrow, wcol := weights.Dims()
	fmt.Println("TEST: rows: ", wrow)
	fmt.Println("TEST: cols: ", wcol)
	var predict mat.Dense
	predict.Mul(testdataMatrix, &weights)
	fmt.Println(predict)
}
