CC=gcc

.PHONY : all

all: sos_filt svd

sos_filt: sos_filt.c
	$(CC) -O3 -fPIC -shared -o accel_sos_filt.so sos_filt.c

svd: svd.c
	$(CC) -O3 -fPIC -shared -o accel_svd.so svd.c

clean:
	rm -f accel_sos_filt.so accel_svd.so
