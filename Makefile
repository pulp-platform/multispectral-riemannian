CC=gcc

sos_filt: sos_filt.c
	$(CC) -O3 -shared -o accel_sos_filt.so sos_filt.c

clean:
	rm -f accel_sos_filt.so
