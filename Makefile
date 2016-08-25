TORCH_INCLUDE = $(shell dirname `which th`)/../include

libnms.so:
	gcc -I$(TORCH_INCLUDE) nms.c -fPIC  -std=c99 -shared -O3 -o $@
clean:
	rm libnms.so
