module adder_4bit_tb;
    reg [3:0] A, B;
    reg Cin;
    wire [3:0] Sum;
    wire Cout;
    
    // Instantiate the adder
    adder_4bit uut (
        .A(A),
        .B(B),
        .Cin(Cin),
        .Sum(Sum),
        .Cout(Cout)
    );
    
    initial begin
        $display("Time\tA\tB\tCin\tSum\tCout");
        $monitor("%0t\t%b\t%b\t%b\t%b\t%b", $time, A, B, Cin, Sum, Cout);
        
        // Test cases
        A = 4'b0011; B = 4'b0101; Cin = 0; #10;  // 3 + 5 = 8
        A = 4'b1111; B = 4'b0001; Cin = 0; #10;  // 15 + 1 = 16 (overflow)
        A = 4'b1010; B = 4'b0110; Cin = 1; #10;  // 10 + 6 + 1 = 17
        
        $finish;
    end
endmodule