CC=gcc

float_as_int_repr: float_as_int_repr.c
	$(CC) -O3 -fPIC -shared -o float_as_int_repr.so float_as_int_repr.c

clean:
	rm -f float_as_int_repr.so
