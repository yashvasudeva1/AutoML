export interface SampleDataset {
  name: string;
  description: string;
  csv: string;
}

export const SAMPLE_DATASETS: SampleDataset[] = [
  {
    name: "Titanic Survival",
    description: "Classification task with mixture of categorical and numeric columns. Perfect for predicting survival.",
    csv: `PassengerId,Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
1,0,3,male,22,1,0,7.25,S
2,1,1,female,38,1,0,71.28,C
3,1,3,female,26,0,0,7.92,S
4,1,1,female,35,1,0,53.1,S
5,0,3,male,35,0,0,8.05,S
6,0,3,male,,0,0,8.45,Q
7,0,1,male,54,0,0,51.86,S
8,0,3,male,2,3,1,21.075,S
9,1,3,female,27,0,2,11.13,S
10,1,2,female,14,1,0,30.07,C
11,1,3,female,4,1,1,16.7,S
12,1,1,female,58,0,0,26.55,S
13,0,3,male,20,0,0,8.05,S
14,0,3,male,39,1,5,31.275,S
15,0,3,female,14,0,0,7.85,S
16,1,2,female,55,0,0,16,S
17,0,3,male,2,4,1,29.125,Q
18,1,2,male,,0,0,13,S
19,0,3,female,31,1,0,18,S
20,1,3,female,22,0,0,7.225,C
21,0,2,male,35,0,0,26,S
22,1,2,male,34,0,0,13,S
23,1,3,female,15,0,0,8.029,Q
24,1,1,male,28,0,0,35.5,S
25,0,3,female,8,3,1,21.075,S
26,1,3,female,38,1,5,31.387,S
27,0,3,male,,0,0,7.225,C
28,0,1,male,19,3,2,263,S
29,1,3,female,,0,0,7.879,Q
30,0,3,male,,0,0,7.896,S
31,0,1,male,40,0,0,27.72,C
32,1,1,female,,1,0,146.52,C
33,1,3,female,,0,0,7.75,Q
34,0,2,male,66,0,0,10.5,S
35,0,1,male,28,1,0,82.17,C
36,0,1,male,42,1,0,52,S
37,1,3,male,,0,0,7.229,C
38,0,3,male,21,0,0,8.05,S
39,0,3,female,18,2,0,18,S
40,1,3,female,14,1,0,11.24,C
41,0,3,female,40,1,0,9.475,S
42,0,2,female,27,1,0,21,S
43,0,3,male,,0,0,7.896,C
44,1,2,female,3,1,2,41.58,C
45,1,3,female,19,0,0,7.879,Q
46,0,3,male,,0,0,8.05,S
47,0,3,male,,1,0,15.5,Q
48,1,3,female,,0,0,7.75,Q
49,0,3,male,,2,0,21.679,C
50,0,3,female,18,1,0,17.8,S
`
  },
  {
    name: "Housing Market Rates",
    description: "Regression dataset with continuous features. Excellent for modeling and predicting real estate pricing.",
    csv: `HouseSize,Rooms,Bathrooms,AgeOfHouse,HasGarage,DistanceToCenter,TaxRate,Price
1500,3,2,10,1,5.2,1.2,320000
1800,4,2.5,5,1,3.1,1.1,395000
1200,2,1,25,0,8.5,1.5,210000
2400,4,3,2,1,1.5,1.0,510000
950,2,1,40,0,10.2,1.6,165000
2100,4,2.5,12,1,4.0,1.2,425000
1650,3,2,8,1,4.8,1.3,340000
1300,3,1.5,18,0,6.5,1.4,260000
2800,5,3.5,1,1,0.8,0.9,620000
1100,2,1,30,0,9.1,1.5,185000
1750,3,2,15,1,5.5,1.2,315000
2000,4,2,7,1,3.8,1.1,410000
1400,3,1.5,22,0,7.0,1.4,245000
2250,4,3,4,1,2.5,1.0,475000
1050,2,1,35,0,9.8,1.6,175000
1900,4,2,11,1,4.2,1.2,380000
1600,3,2,9,1,5.0,1.3,330000
1250,2,1.5,20,0,6.9,1.4,230000
2600,5,3,3,1,1.2,1.0,560000
1150,2,1,28,0,8.8,1.5,195000
1700,3,2,13,1,5.4,1.2,325000
1950,4,2,6,1,3.6,1.1,405000
1350,3,1.5,24,0,7.5,1.4,225000
2300,4,3,2,1,2.0,1.0,490000
1000,2,1,38,0,10.0,1.6,170000
1850,3,2,12,1,4.4,1.2,365000
1550,3,2,10,1,5.1,1.3,310000
1450,3,1.5,19,0,6.8,1.4,250000
2500,5,3.5,2,1,1.0,0.9,590000
1080,2,1,32,0,9.4,1.5,180000
`
  },
  {
    name: "Iris Species",
    description: "Classic multi-class classification dataset with physical measurements of three iris flower species.",
    csv: `SepalLength,SepalWidth,PetalLength,PetalWidth,Species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
7.6,3.0,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica
`
  }
];
