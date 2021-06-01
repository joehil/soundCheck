package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
//	"strings"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

var neural *deep.Neural
var neuralConfig deep.Config   


func main() {
// Get commandline args
	if len(os.Args) > 1 {
        	a1 := os.Args[1]
        	if a1 == "train" {
			setup()
			train()
			os.Exit(0)
        	}
                if a1 == "predict" {
			setup()
                        predict()
			os.Exit(0)
                }
		fmt.Println("parameter invalid")
		os.Exit(-1)
	}
	if len(os.Args) == 1 {
		myUsage()
	}
}

func setup(){
	neuralConfig.Inputs = 13
	neuralConfig.Layout = []int{8, 3}
	neuralConfig.Activation = deep.ActivationTanh
	neuralConfig.Mode = deep.ModeMultiClass
	neuralConfig.Weight = deep.NewNormal(1, 0)
	neuralConfig.Bias = true
}

func train(){
	rand.Seed(time.Now().UnixNano())

	data, err := load("./wine.data")
	if err != nil {
		panic(err)
	}

	for i := range data {
		deep.Standardize(data[i].Input)
	}
	data.Shuffle()

	fmt.Printf("have %d columns\n", len(data[0].Input))
	fmt.Printf("have %d entries\n", len(data))

	neural = deep.NewNeural(&neuralConfig)

	//trainer := training.NewTrainer(training.NewSGD(0.005, 0.5, 1e-6, true), 50)
	//trainer := training.NewBatchTrainer(training.NewSGD(0.005, 0.1, 0, true), 50, 300, 16)
	//trainer := training.NewTrainer(training.NewAdam(0.1, 0, 0, 0), 50)
	trainer := training.NewBatchTrainer(training.NewAdam(0.1, 0, 0, 0), 50, len(data)/2, 12)
	//data, heldout := data.Split(0.5)
	trainer.Train(neural, data, data, 5000)

	fmt.Println(neural.Predict(data[0].Input))
	fmt.Println(neural.Predict(data[5].Input))

	dump, err := neural.Marshal()

	f, err := os.Create("./sound.network")
    	defer f.Close()
	n2, err := f.Write(dump)
	fmt.Printf("wrote %d bytes\n", n2)
	f.Sync()
}

func predict(){
	data := make([]float64, 13)

	f, err := os.Open("./sound.network")
	defer f.Close()

	stat, err := f.Stat()
	dump := make([]byte, stat.Size())

	n1, err := f.Read(dump)
	check(err)

	fmt.Printf("read %d bytes\n", n1)

	argsWithoutProg := os.Args[1:]
	fmt.Println(argsWithoutProg)

	for i := 0; i < 13; i++ {
		data[i], err = strconv.ParseFloat(os.Args[i+2], 64)
	}
	deep.Standardize(data)

	newneural, err := deep.Unmarshal(dump)
	check(err)

	fmt.Println(newneural.Predict(data))
}

func load(path string) (training.Examples, error) {
	f, err := os.Open(path)
	defer f.Close()
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(bufio.NewReader(f))

	var examples training.Examples
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		examples = append(examples, toExample(record))
	}

	return examples, nil
}

func toExample(in []string) training.Example {
	res, err := strconv.ParseFloat(in[0], 64)
	if err != nil {
		panic(err)
	}
	resEncoded := onehot(3, res)
	var features []float64
	for i := 1; i < len(in); i++ {
		res, err := strconv.ParseFloat(in[i], 64)
		if err != nil {
			panic(err)
		}
		features = append(features, res)
	}

	return training.Example{
		Response: resEncoded,
		Input:    features,
	}
}

func onehot(classes int, val float64) []float64 {
	res := make([]float64, classes)
	res[int(val)-1] = 1
	return res
}

func myUsage() {
     fmt.Printf("Usage: %s argument\n", os.Args[0])
     fmt.Println("Arguments:")
     fmt.Println("train         Train the neural network")
     fmt.Println("predict       Predict the outcome")
}

func check(e error) {
    if e != nil {
        panic(e)
    }
}

