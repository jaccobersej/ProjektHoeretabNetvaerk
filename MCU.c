#include <avr/io.h>
#include <util/delay.h>
#include <avr/interrupt.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef F_CPU //hvis F_CPU  (hastighed for MCU, (konstant / her til brug i udregninger)
#define F_CPU 8000000UL  //så definer den
#endif
#define BAUDRATEVAL 9600 //Definer Baud raten. Vores MCU har en baud rate på 9600
#define BIT(x) (1 << (x)) //
#define SETBITS(x,y) ((x) |= (y))
#define CLEARBITS(x,y) ((x) &= (~(y)))
#define SETBIT(x,y) SETBITS((x), (BIT((y))))
#define CLEARBIT(x,y) CLEARBITS((x), (BIT((y))))
#define BITVAL(x,y) (((x)>>(y)) & 1)

void tx_serial_number(uint16_t n);
void tx_serial(volatile char data[]);

volatile char uart_buffer[100];
volatile uint16_t counter = 0;
volatile uint8_t u_index = 0;


void init_seriel(){
    uint16_t ubrr0;     //initialize 16-bit integer
    ubrr0 = (((F_CPU / (BAUDRATEVAL * 16UL))) - 1); //Definer tallet som værende frekvensen på cpu'en delt med baud raten gange 16UL og bagefter minus det hele med 1
    UBRR0H = (unsigned char) (ubrr0>>8);	//Gem de 8 mest betydende bits fra vores 16-bit integer
    UBRR0L = (unsigned char) (ubrr0);		//gem de 8 mindst betydende bits fra vores 16-bit integer
    UCSR0C = (1<<UCSZ00) | (1<<UCSZ01);     //8 bit, 1 stop, no parity
    UCSR0B = (1<<TXEN0) | (1<<RXCIE0) | (1<<RXEN0); // Enable  transmitter, receiver rx interrupt
}

ISR(USART_RX_vect){
        uart_buffer[u_index++] = UDR0;
}


void timer0_init(){
//timer 0 initialize
    TCCR0B = (1<<CS02); //prescaler = 256
    TIFR0 = (1<<TOV0); //clear pending interrupts
    TIMSK0  = (1<<TOIE0); //Enable timer 0 overflow
}

ISR(TIMER0_OVF_vect){
        counter++;
}

void timer1_init(void){
//timer 1 initialize
    TCCR1B |= (1<<CS10); //prescaler = 1
}


void tx_serial(volatile char data[]){	//transmitter serial med input data array, hvor de kan være vilkårlig type.
    uint8_t i = 0;
    while(data[i] != 0)
    {
        while (!( UCSR0A & (1<<UDRE0)));
        UDR0 = data[i];
        i++;
    }
}

void tx_serial_number(uint16_t n){
    char string[8];
    itoa(n, string,10); //10 is radix
    tx_serial(string);
}


void init(){
    cli();  //Disable interrupts while setting registers
    DDRD = 0xFF; //sæt alle port PD til Transmit/output, men de kan stadig læses
    init_seriel();
    timer0_init();
    timer1_init();
    sei(); //global interrupt enable, global disable is: cli();
}


int main(void)
{
    init();

    uint16_t Hz_values[6] ={150, 5000, 7500, 10000, 15000, 20000}; //standard værdier

    while(1)
    {
        volatile char stopSign = ";";
        if(uart_buffer[0] != 0) {
            uint8_t j;
            for (j = 0; j < sizeof(uart_buffer); j++) {
                if (uart_buffer[j] == stopSign) {

                    _delay_ms(50);

                    char *input = uart_buffer; //gemmer bufferen som en pointer

                    char *firstPart = strtok(input, " "); //input skal være en pointer i denne funktion
                    char *secondPart = strtok(NULL, " ");
                    uint16_t firstInt = atoi(firstPart);
                    uint16_t secondInt = atoi(secondPart);
                    uint16_t space = (secondInt - firstInt)/5;
                    uint8_t p;
                    for (p = 0; p < 6; p++) {
                        Hz_values[p] = firstInt + p * space;
                    }
                    _delay_ms(100);
                    uint8_t k;
                    for(k = 0; k < sizeof uart_buffer; k++) {
                        uart_buffer[k] = 0;
                    }
                    u_index = 0;
                    _delay_ms(50);
                    break;
                }
            }
            SETBIT(PORTD, PD4);
        }

        if ((PIND & (1<<PD7)) && (PIND & (1<<PD4))) {
            CLEARBIT(PORTD, PD4);
            SETBIT(PORTD, PD2);
            _delay_ms(1000);
            int time_spent = 0;
            uint8_t i = 0;
            for (i = 0; i < 6; i++) {
                OCR1A = (int) (4000000/Hz_values[i]);  //Ny Compare Register
                _delay_ms(2000);
                counter = 0;
                while ((counter < 246) && (~PIND & (1<<PD3))) { //246 = 2 sekunder
                    SETBIT(PORTD, PD5);
                    while((TIFR1 & (1<<OCF1A)) == 0);
                    TCNT1 = 0;
                    TIFR1 |= (1<<OCF1A);
                    CLEARBIT(PORTD, PD5);
                    while((TIFR1 & (1<<OCF1A)) == 0);
                    TCNT1 = 0;
                    TIFR1 |= (1<<OCF1A);
                }
                tx_serial_number(counter);
                tx_serial(";");
                _delay_ms(2000);
                counter = 0;
                while ((counter < 246) && (~PIND & (1<<PD3))) {
                    SETBIT(PORTD, PD6);
                    while ((TIFR1 & (1 << OCF1A)) == 0);
                    TCNT1 = 0;
                    TIFR1 |= (1 << OCF1A);
                    CLEARBIT(PORTD, PD6);
                    while ((TIFR1 & (1 << OCF1A)) == 0);
                    TCNT1 = 0;
                    TIFR1 |= (1 << OCF1A);
                }
                tx_serial_number(counter);
                tx_serial(";");
            }
        } else {
            CLEARBIT(PORTD, PD2);
        }
    }
    return 0;
}