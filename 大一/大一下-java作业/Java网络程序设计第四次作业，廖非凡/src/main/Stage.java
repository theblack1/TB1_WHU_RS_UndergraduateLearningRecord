package main;

import InterFaces.Stuff;
import person.Ruler;

public abstract class Stage implements Stuff {
	public String name = "";
	public Ruler boss = new Ruler();

	public Stage() {
	}

}
