CC=arm-linux-gnueabi-g++
CFLAGS=

all:
	$(CC) $(CFLAGS) main.cpp -o npu_tester -I/usr/lib/arm-linux-gnueabi/include -L/usr/lib/arm-linux-gnueabi/lib -lcnpy -lz
