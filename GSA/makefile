all:
	nvcc -G --ptxas-options=-v *.c *.cu -O3 -o m

clean:
	rm -f m && rm -f *.o

run:
	./m e801600 100000 5 176828
