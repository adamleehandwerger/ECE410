.PHONY: all clean run

all: obj_dir/Vadder_4bit

obj_dir/Vadder_4bit: adder_4bit.v sim_main.cpp
	verilator -Wall --cc adder_4bit.v --exe sim_main.cpp
	make -C obj_dir -f Vadder_4bit.mk Vadder_4bit

run: all
	./obj_dir/Vadder_4bit

clean:
	rm -rf obj_dir