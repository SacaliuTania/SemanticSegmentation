# English Version

# 1. Project Design and Implementation
## 1.1 Chosen Experimental Method
The experimental method is a software approach based on a hybrid pipeline. The project is structured across two programming environments, Python and C++, meaning the training part is completely decoupled from the execution part. Even though these two systems are independent, they communicate asynchronously. The Python program handles model learning and training. This module must properly manage the chosen dataset (CityScapes) and correctly perform class mapping.  
The C++ program uses the exported model for inference optimization. This system runs purely on CPU without requiring hardware dependencies.

## 1.2 Chosen Solution
DeepLabV3 introduces Atrous convolutions to solve resolution issues. This architecture routes images through multiple structural layers: Input -> Atrous Convolution -> Score Map -> Bilinear Interpolation -> Fully Connected CRF -> Output. The network successfully captures both massive objects and smaller or distant objects within the image. It uses pre-trained weights on the COCO and PascalVOC datasets because pre-trained weights for the CityScapes dataset were unavailable.  
Due to VRAM limitations, images were significantly downscaled, and a batch size of 2 was used.

## 1.3 Implemented Algorithms
### Label Mapping
CityScapes masks use official IDs from 0 to 33 for pixel labeling. The label mapping algorithm maps the CityScapes indices into 19 consecutive numbers from 0 to 18, representing the model's 19 classes.

### Transfer Learning Algorithm
Pre-trained weights from COCO and PascalVOC models were loaded. These models contain 21 classes, whereas CityScapes uses 19. New convolutional layers were randomly initialized and adapted to the 19-class topology, allowing all parameters to be modified at optimizer-controlled rates.

### Bilinear Interpolation
This algorithm estimates the exact value of a pixel when it does not align precisely with the original pixel grid. The 4 closest known pixels are identified, and their weighted average is calculated by performing horizontal interpolation followed by vertical interpolation.

### Data Preprocessing
Images are loaded and read using OpenCV's `imread` function, stored in memory in HWC (Height, Width, Channel) format with BGR layout. Images are then resized to 512 x 256 pixels.  
Because the model expects the RGB standard in a planar format (RRR...R, GGG...G, BBB...B), a Blob (Binary Large Object) is manually constructed by allocating a vector in memory.

### Inference
Raw C++ data is transformed into a LibTorch object. The model is loaded into memory and switched to evaluation mode, which disables Dropout and Batch Normalization layers to prevent altering predictions.

### Post-processing
The network output is a 3D tensor of dimensions 19 x 256 x 512. Each pixel is assigned a probability score for each of the 19 dataset classes. The program loops vertically across the 19 channels to retain only the class index with the highest score, turning the result into a simple 2D matrix.  
A new OpenCV matrix is created, where for each pixel in the ArgMax matrix, the class ID is read and replaced with its corresponding RGB value from the color lookup table. Three images are generated: the raw segmentation result and an overlay image, concatenated horizontally into a single window.

# 2. Experimental Results
Available hardware resources were limited; therefore, a controlled training strategy was chosen. Images were resized to W = 256 and H = 512 pixels, and the number of batches per epoch was restricted to 200 to keep training times down to a few minutes. The loss function drops reasonably during the first few epochs, proving that the DeepLab network correctly adapts its weights for the CityScapes dataset. For ideal results, the epoch count should be extended to 20, and batch limits removed.  
The model begins correctly segmenting large and obvious classes when the loss falls below 1.0. A loss of 0.7 after 10 epochs yielded good results, though it did not fully reach the desired semantic segmentation level.  
Subsequently, epochs were increased from 10 to 20, and the batch size grew from 200 to 500. By epoch 5 of this new experiment, loss dropped significantly to 0.54. Values between 0.4 and 0.6 yield near-ideal results. An optimal model achieves a loss lower than 0.2. Experimentally, the functionality of the Python, PyTorch, and C++ pipeline is fully proven, yielding expected results.

### Color Mapping Table
<img width="546" height="499" alt="image" src="https://github.com/user-attachments/assets/343f224b-90b0-4068-a662-ce1de5099afd" />


# Varianta în română

# 1. Proiectare și implementare
## 1.1 Metoda experimentală aleasă
Metoda experimentală este o metodă software care se bazează pe un pipeline hibrid. Proiectul este structurat pe două medii de programare Python și C++. Astfel, partea de antrenare este complet separată de partea de execuție. Chiar dacă aceste două sisteme sunt independente comunică asincron. Programul scris în Python se ocupă de învățarea și antrenarea modelului. Acest modul al proiectului trebuie să gestioneze corect setul de date ales (CityScapes) și trebuie să realizeze corect maparea claselor.  
Programul în C++ folosește modelul exportat pentru optimizarea inferenței. Acest sistem rulează doar pe CPU, fără a necesita dependințe hardware.

## 1.2 Soluția aleasă
DeepLabV3 introduce convoluții Atrous pentru a rezolva problemele de rezoluție. Această arhitectură face ca imaginea să treacă prin mai multe straturi arhitecturale: Input -> AtrousConvolution -> Score Map -> Interpolare biliniară -> Fully Connected CRF -> Output. Rețeaua reușește să capteze atât obiectele masive, cât și obiectele mai mici sau la distanțe mai mari în imagine. Rețeaua folosește ponderi preantrenate pe seturile de date COCO și PascalVOC, deoarece nu s-au putut găsi ponderi preantrenate pentru setul de date CityScapes.  
Din cauza limitărilor de memorie VRAM, s-au scalat imaginile reducându-le semnificativ dimensiunea și s-a utilizat un batch de dimensiune 2.

## 1.3 Algoritmii implementați
### Label mapping
Măștile din setul de date CityScapes folosesc ID-uri oficiale de la 0 la 33 pentru etichetarea pixelilor. Algoritmul de label mapping ajută ca indicii din CityScapes să fie reprezentați ca 19 numere consecutive de la 0 la 18, reprezentând cele 19 clase ale modelului.

### Algoritmul de învățare prin transfer
Au fost încărcate ponderile preantrenate ale modelelor COCO și PascalVOC. Aceste modele conțin 21 de clase, dar CityScapes folosește 19. Straturile convoluționale noi au fost reinițializate aleatoriu, adaptate topologiei de 19 clase. Astfel, algoritmul permite modificarea tuturor parametrilor cu viteze controlate de optimizator.

### Interpolare biliniară
Acest algoritm a fost folosit pentru estimarea corectă a valorii unui pixel dintr-o poziție în care nu se aliniază corect cu grila originală de pixeli. Sunt identificați cei mai apropiați 4 pixeli cunoscuți și este calculată media lor ponderată. Se realizează mai întâi o interpolare pe axa orizontală, urmată de o interpolare pe axa verticală.

### Preprocesarea datelor
Se încarcă și se citește imaginea folosind funcția imread din OpenCV. Imaginea este stocată în memorie în format HWC (Height, Width, Channel), BGR. Urmează ca imaginea să fie redimensionată la 512 x 256 de pixeli.  
Modelul folosește standardul RGB, așadar se convertește în formatul corespunzător. Rețelele neuronale au nevoie ca pixelii să fie în formatul planar. Modelul primește formatul RGB, dar așteaptă formatul RRR...R, GGG...G, BBB...B. Se construiește manual un blob (Binary Large Object), prin alocarea unui vector în memorie.

### Inferența
Datele brute din C++ sunt transformate într-un obiect LibTorch. Se încarcă modelul în memorie și trece de starea de evaluare. În această stare de evaluare sunt dezactivate straturile de tip Dropout sau Batch Normalization, deoarece acestea pot altera predicția.

### Postprocesarea
Output-ul rețelei este un tensor 3D de dimensiuni 19 x 256 x 512. Fiecărui pixel îi este atribuit un scor de probabilitate pentru fiecare dintre cele 19 clase ale setului de date. Se parcurg cele 19 canale pe verticală și se reține doar numărul clasei care a obținut cel mai mare scor. Rezultatul devine o matrice simplă 2D.  
Se creează o nouă matrice OpenCV și pentru fiecare pixel din matricea ArgMax se citește ID-ul clasei și se înlocuiește cu valoarea RGB corespunzătoare din tabelul culorilor. Se obțin 2 imagini: imaginea rezultat în urma segmentării și o imagine cu overlay. Cele trei imagini sunt concatenate pe orizontală într-o singură fereastră.

# 2. Rezultate experimentale
Resursele hardware disponibile sunt limitate. Astfel, pentru antrenare am ales o strategie controlată. Imaginile au fost redimensionate la W = 256 și H = 512 pixeli, iar numărul de batch-uri pe epocă a fost redus la 200 pentru a asigura un timp de antrenare de câteva minute. Funcția de cost scade rezonabil în primele epoci, demonstrând astfel că rețeaua DeepLab își adaptează corect ponderile pentru imaginile din setul de date CityScapes. Pentru rezultate ideale, numărul de epoci ar trebui extins la 20 și eliminată limita de batch-uri.  
Modelul începe să segmenteze corect clasele mari și evidente atunci când loss-ul are o valoare mai mică de 1.0. Atunci când modelul a avut un loss de 0.7 după 10 epoci, rezultatul a fost bun, însă nu a atins nivelul de segmentare semantică așteptat și dorit.  
Ulterior, a fost modificat numărul de epoci de la 10 la 20, iar dimensiunea batch-ului a fost crescută de la 200 la 500. În urma noului experiment, până la epoca 5, loss-ul a scăzut semnificativ atingând o valoare bună de 0.54. În intervalul 0.4-0.6 modelul dă rezultate mai bune, aproape ideale.


Maparea culorilor


<img width="589" height="488" alt="image" src="https://github.com/user-attachments/assets/fbf77dca-bf60-4156-877d-8bd7c18ece59" />
