CC=gcc

.PHONY : all

sos_filt: sos_filt.c
	$(CC) -O3 -fPIC -shared -o accel_sos_filt.so sos_filt.c

svd: svd.c
	$(CC) -O3 -fPIC -shared -o accel_svd.so svd.c

all: sos_filt svd

clean:
	rm -f accel_sos_filt.so
	rm -f accel_svd.so
