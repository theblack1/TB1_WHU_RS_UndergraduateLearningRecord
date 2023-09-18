
package ch04.usefulclass;

public class ObjectEqualDemo102 {

	public static void main(String[] args) {

		Person102 person1 = new Person102("Tony", 20);
		Person102 person2 = new Person102("Tom", 20);

		System.out.println(person1 == person2); // false
		System.out.println(person1.equals(person2));// true

	}
}
