#!/usr/bin/perl
use 5.16.0;
use warnings FATAL => 'all';

use Test::Simple tests => 16;
use IO::Handle;

system("(cd tools && make clean && make) > /dev/null");

ok(-e "report.txt", "report.txt exists");
ok(-e "graph.png", "graph.png exists");

sub make_data {
    my ($nn) = @_;
    system("rm -f data0.dat data1.dat");
    system("tools/gen-input $nn data0.dat");
}

sub check_errors {
    my ($errs) = @_;
    chomp $errs;
    if ($errs ne "") {
        $errs =~ s/^/# /mg;
        warn $errs;
        return 0;
    }
    else {
        return 1;
    }
}

sub check_output {
    my ($outp, $pp) = @_;
    chomp $outp;

    my @lines = split /\n/, $outp;
    @lines = sort {
        my $aa = $a;
        $aa =~ s/:.*$//;
        my $bb = $b;
        $bb =~ s/:.*$//;
        $aa <=> +$bb;
    } @lines;

    my $good = 0;
    my $prev = 0.0;
    my $nums = 0;

    for my $ll (@lines) {
        unless ($ll =~ /\d+:\s+start\s+(\d+.\d+),\s+count\s+(\d+)/) {
            say "# line didn't match pattern";
            say "# $ll";
            say join("\n", @lines);
            last;
        }

        my $start = $1;
        my $count = $2;

        if ($start < $prev) {
            say "# range start not in increasing order";
            say "# $start < $prev";
            say join("\n", @lines);
            last;
        }

        $prev  = $start;
        $nums += $count;
        $good += 1;
    }

    return ($good, $nums);
}


sub run_test {
    my ($nn, $pp) = @_;

    make_data($nn);
    my $outp = `timeout -k 10 10 ./tssort $pp data0.dat data1.dat`;
    my ($good, $nums) = check_output($outp);
    ok($good == $pp, "ssort $pp ($nn) - partition order");
    ok($nums == $nn, "ssort $pp ($nn) - partition counts");

    my $sorted = `tools/check-sorted data1.dat`;
    ok(check_errors($sorted), "ssort $pp ($nn) - sorting");
}

run_test(10, 1);
run_test(1000, 1);
run_test(1000, 4);
run_test(1899, 16);

sub valgrind {
    make_data(100);
    my $vg = "valgrind -q --leak-check=full --log-file=valgrind.out";
    system(qq{timeout -k 10 10 $vg ./tssort 2 data0.dat data1.dat > /dev/null});
    my $errs = `cat -n valgrind.out`;
    return check_errors($errs);
}

sub clang_check {
    my $errs = `clang-check -analyze *.c -- 2>&1`;
    return check_errors($errs);
}

ok(clang_check(), "clang check");
ok(valgrind(), "valgrind");

system("rm -f data0.dat data1.dat");
