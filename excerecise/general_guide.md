# General program guide
> "I will help you bear this burden Frodo Baggins, as long as it is yours to bear." Gandalf the Grey

## code tags
There are several comments in code with specific purpose. Examples are given to each cathegory of comment.
* **plain comment** - general description of the piece of code

    * ```C
    // useful description of code here
    ```

* **block comment** - code section divider usually with row of "="
    * ```C
    // === important section ===
    // == less important section ==
    /* another section type*/
    ```
* **NOTE** - noteworthy feature in the code, have a closer look
    * ```C
    //NOTE - the function call is looking suspicious
    ```

* **TASK** - perform the task given and see the result, to better understand what is happening
    * ```C
    //TASK - comment this to see the difference
    ```

* **BUG** - warns about the high chance of making a mistake in this area, error prone area
	* ```C
	//BUG - check the memory allocation
	```
* **TODO** - helps spot places for extension and pending code enhancments
	* ```C
	//TODO - implement more rigorous checking
	```

## what to do when lost
> "If In Doubt, Meriadoc, Always Follow Your Nose!" Gandalf the Grey

* think abou the problem
	* dissassemble the issue into smaler pieces
	* solve the smaller pieces
* search the relevant resources
	* refer to the literature for help (available from presentations)
	* refer to the documentation (docs.nvidia.com/cuda, developer.nvidia.com/search)
* "google" the answer
	* use the propper formulation
	* specify the query by exact match "", substraction - , etc.	
* ask me personally/remotely
	* ask per email
		* expect some delay for answer
	* make an appointment per email 
		* for more complicated problems that are hard to explain